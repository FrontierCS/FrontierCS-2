"""Trusted evaluation of one serialised policy. Runs as ``python -I -B``.

The environment, its command sampling, physics, reward terms and all random
streams come from the pristine tree given by ``--repo`` (hash-verified by the
parent immediately before launch). The policy arrives as a ``jax.export``
artifact and is only ever called as a tensor function; no submitted Python is
imported here.

Measurement (identical for both sides): ``envs`` parallel environments are
reset from ``split(PRNGKey(eval_seed), envs)`` and stepped ``steps`` control
steps with the environment's own command resampling and no automatic reset.
An environment is *alive* at a step if it has not terminated at or before that
step. For each environment the upstream step reward ``state.reward`` and the
upstream per-term values ``state.metrics["reward/<term>"]`` are summed over its
alive steps.

* An episode metric is the mean over environments of those per-environment
  sums (falling early lowers it).
* A step metric is the sum over all environments of those per-environment sums
  divided by the total number of alive steps, so a policy is neither rewarded
  for falling early nor penalised for surviving longer.
"""

import argparse
import json
import os
import sys


def write_report(path, document):
    body = json.dumps(document, allow_nan=False, sort_keys=True).encode("ascii")
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(body)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--overrides", required=True, help="JSON {dotted.config.key: value}")
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--observation-size", type=int, required=True)
    parser.add_argument("--action-size", type=int, required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--episode-metrics", required=True, help="comma-separated names")
    parser.add_argument("--step-metrics", required=True, help="comma-separated names")
    parser.add_argument("--report", required=True)
    opts = parser.parse_args()
    episode_metrics = tuple(filter(None, opts.episode_metrics.split(",")))
    step_metrics = tuple(filter(None, opts.step_metrics.split(",")))
    names = episode_metrics + step_metrics
    if len(set(names)) != len(names) or not names:
        raise ValueError("metric names must be unique and non-empty")

    def measured(state, name):
        # ``episode_reward`` is the environment's own total step reward.
        return state.reward if name == "episode_reward" else state.metrics["reward/" + name]

    import jax
    import jax.numpy as jp
    from jax import export

    sys.path.insert(0, opts.repo)
    from mujoco_playground import registry

    report = {"gpu": jax.devices()[0].platform == "gpu", "env_name": opts.env_name,
              "eval_seed": opts.eval_seed, "envs": opts.envs, "steps": opts.steps}

    # Trusted environment. A failure here is the judge's, never the policy's.
    config = registry.get_default_config(opts.env_name)
    for dotted, value in json.loads(opts.overrides).items():
        node = config
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError("unknown environment config key: " + dotted)
        node[leaf] = value
    env = registry.load(opts.env_name, config=config)
    if env.action_size != opts.action_size:
        raise ValueError("unexpected environment action size")
    reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)
    master = jax.random.PRNGKey(opts.eval_seed)
    initial = jax.jit(reset)(jax.random.split(master, opts.envs))
    if tuple(initial.obs["state"].shape) != (opts.envs, opts.observation_size):
        raise ValueError("unexpected environment observation size")
    missing = [name for name in names
               if name != "episode_reward" and "reward/" + name not in initial.metrics]
    if missing:
        raise KeyError("environment does not report: " + ",".join(missing))

    # Untrusted policy artifact. A failure from here on is attributed to it.
    try:
        if os.path.getsize(opts.artifact) > opts.max_artifact_bytes:
            raise ValueError("policy artifact exceeds its size limit")
        with open(opts.artifact, "rb") as stream:
            policy = export.deserialize(bytearray(stream.read()))
        inputs = [(tuple(a.shape), str(a.dtype)) for a in policy.in_avals]
        outputs = [(tuple(a.shape), str(a.dtype)) for a in policy.out_avals]
        if inputs != [((opts.envs, opts.observation_size), "float32"), ((opts.envs, 2), "uint32")]:
            raise ValueError("policy artifact has the wrong input signature")
        if outputs != [((opts.envs, opts.action_size), "float32")]:
            raise ValueError("policy artifact has the wrong output signature")

        def control_step(carry, _):
            state, rng, alive, sums, count, finite = carry
            rng, key = jax.random.split(rng)
            action = policy.call(state.obs["state"], jax.random.split(key, opts.envs))
            # Only environments that are still alive are judged. After its first
            # termination an environment keeps stepping without a reset, so its
            # state and the action computed from it may be non-finite; they are
            # excluded from every measurement and must not reject the policy.
            finite = finite & jp.all(jp.where(alive[:, None], jp.isfinite(action), True))
            state = step(state, action)
            alive = alive & (state.done < 0.5)
            sums = {name: sums[name] + jp.where(alive, measured(state, name), 0.0)
                    for name in names}
            count = count + alive.astype(jp.float32)
            return (state, rng, alive, sums, count, finite), None

        @jax.jit
        def rollout(state, rng):
            zeros = jp.zeros(opts.envs)
            carry = (state, rng, jp.ones(opts.envs, dtype=bool),
                     {name: zeros for name in names}, zeros, jp.array(True))
            (_, _, alive, sums, count, finite), _ = jax.lax.scan(
                control_step, carry, None, length=opts.steps)
            alive_steps = jp.sum(count)
            results = {name: jp.mean(sums[name]) for name in episode_metrics}
            results.update({name: jp.sum(sums[name]) / jp.maximum(alive_steps, 1.0)
                            for name in step_metrics})
            return results, finite, jp.mean(alive.astype(jp.float32)), alive_steps

        means, finite, survived, alive_steps = rollout(initial, master)
        values = {name: float(value) for name, value in means.items()}
        if float(alive_steps) < 1.0:
            raise ValueError("no environment survived a single control step")
        if not bool(finite):
            raise ValueError("policy produced a non-finite action")
        if not all(value == value and abs(value) != float("inf") for value in values.values()):
            raise ValueError("policy produced non-finite episode measurements")
    except Exception as error:  # noqa: BLE001 - attributed to the artifact
        print(f"evaluate_policy: policy rejected: {type(error).__name__}: {error}", flush=True)
        report["invalid"] = type(error).__name__
        report["invalid_detail"] = str(error)[:200]
        write_report(opts.report, report)
        return
    report["metrics"] = values
    report["finite_actions"] = True
    report["survived_fraction"] = float(survived)
    report["alive_step_fraction"] = float(alive_steps) / (opts.envs * opts.steps)
    write_report(opts.report, report)


if __name__ == "__main__":
    main()

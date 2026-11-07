"""Public self-check: train with ``frontier_solution`` and measure the policy.

This is a convenience for your own GPU runner, not the grader. It follows the
same steps the grader documents in README.md: call
``frontier_solution.train.train(seed, deadline_seconds)``, serialise the
returned policy with ``jax.export`` as a batched function of the actor
observation, and roll it out in this repository's environments with the
measurement defined in the README's Scoring section. The grader uses its own
unmodified copy of the environment code and its own private seeds, so a local
edit to the environment changes what this script measures but not what the
grader measures.

Example (on the GPU runner, from the repository root):

  /opt/frontier/task-runtime /opt/frontier/runtime-python frontier_check.py \
      --seed 0 --conditions flat,push,rough
"""

import argparse
import json
import time

CONDITIONS = {
    "flat": ("Go1JoystickFlatTerrain", {}),
    "push": ("Go1JoystickFlatTerrain", {"pert_config.enable": True}),
    "rough": ("Go1JoystickRoughTerrain", {}),
}
EPISODE_METRICS = ("episode_reward", "tracking_lin_vel", "tracking_ang_vel")
STEP_METRICS = ("energy", "action_rate")
OBSERVATION_SIZE = 48
ACTION_SIZE = 12


def measure(policy, env_name, overrides, eval_seed, envs, steps):
  """Rolls out an exported batched policy; see README.md, Scoring."""
  import jax
  import jax.numpy as jp

  from mujoco_playground import registry

  config = registry.get_default_config(env_name)
  for dotted, value in overrides.items():
    node = config
    *parents, leaf = dotted.split(".")
    for part in parents:
      node = node[part]
    node[leaf] = value
  env = registry.load(env_name, config=config)
  step = jax.vmap(env.step)
  names = EPISODE_METRICS + STEP_METRICS

  def measured(state, name):
    if name == "episode_reward":
      return state.reward
    return state.metrics["reward/" + name]

  def control_step(carry, _):
    state, rng, alive, sums, count = carry
    rng, key = jax.random.split(rng)
    action = policy.call(state.obs["state"], jax.random.split(key, envs))
    state = step(state, action)
    alive = alive & (state.done < 0.5)
    sums = {
        name: sums[name] + jp.where(alive, measured(state, name), 0.0)
        for name in names
    }
    return (state, rng, alive, sums, count + alive.astype(jp.float32)), None

  @jax.jit
  def rollout(state, rng):
    zeros = jp.zeros(envs)
    carry = (state, rng, jp.ones(envs, dtype=bool),
             {name: zeros for name in names}, zeros)
    (_, _, alive, sums, count), _ = jax.lax.scan(
        control_step, carry, None, length=steps
    )
    alive_steps = jp.maximum(jp.sum(count), 1.0)
    out = {name: jp.mean(sums[name]) for name in EPISODE_METRICS}
    out.update({name: jp.sum(sums[name]) / alive_steps for name in STEP_METRICS})
    out["survived_fraction"] = jp.mean(alive.astype(jp.float32))
    return out

  master = jax.random.PRNGKey(eval_seed)
  initial = jax.jit(jax.vmap(env.reset))(jax.random.split(master, envs))
  return {name: float(value) for name, value in rollout(initial, master).items()}


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--eval-seed", type=int, default=1)
  parser.add_argument("--conditions", default="flat,push,rough")
  parser.add_argument("--envs", type=int, default=2048)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--budget-seconds", type=float, default=720.0)
  opts = parser.parse_args()

  import jax
  import jax.numpy as jp
  from jax import export

  from frontier_solution.train import train

  began = time.monotonic()
  policy = train(opts.seed, opts.budget_seconds)
  train_seconds = time.monotonic() - began

  def batched(observations, keys):
    return jp.asarray(jax.vmap(policy)(observations, keys), dtype=jp.float32)

  exported = export.export(jax.jit(batched))(
      jax.ShapeDtypeStruct((opts.envs, OBSERVATION_SIZE), jp.float32),
      jax.ShapeDtypeStruct((opts.envs, 2), jp.uint32),
  )
  assert tuple(exported.out_avals[0].shape) == (opts.envs, ACTION_SIZE)
  restored = export.deserialize(exported.serialize())
  result = {
      "devices": [str(device) for device in jax.devices()],
      # The graded budget is the duration of the train() call alone.
      "train_seconds": train_seconds,
      "budget_seconds": opts.budget_seconds,
      "export_seconds": time.monotonic() - began - train_seconds,
  }
  for name in filter(None, opts.conditions.split(",")):
    env_name, overrides = CONDITIONS[name]
    result[name] = measure(
        restored, env_name, overrides, opts.eval_seed, opts.envs, opts.steps
    )
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()

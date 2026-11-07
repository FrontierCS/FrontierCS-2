"""Trusted wrapper around one training run. Runs as ``python -I -B``.

Runs under the shared isolate launcher (runtime.ChildRunner.run_sandboxed). Stage markers are
written to the original stdout, the judge's pipe (the process's own stdout is moved to stderr
first), each limited by the parent:

  I  pristine site-packages (JAX, Brax, MuJoCo) are imported. Nothing submitted
     has run. Process start and these imports are infrastructure time.
  T  the given tree is on ``sys.path`` and ``frontier_solution.train`` is
     imported; ``train(seed, deadline_seconds)`` is called next. The training
     budget is the wall-clock time from this marker to the next one, and
     ``deadline_seconds`` is that same budget.
  D  ``train`` has returned. What remains is tracing ``jax.vmap(policy)`` and
     serialising it with ``jax.export`` as ``(obs[N, obs_size] f32, keys[N, 2]
     u32) -> action[N, action_size] f32``, under its own limit.
  E  the artifact is in place; the process exits next (within the same limit).

The artifact is StableHLO: a pure tensor program with the trained parameters
embedded as constants. It is the only thing the evaluation ever takes from this
process. For a candidate this process is untrusted after ``I`` (submitted code
could write the later markers itself; doing so early only shortens its own
remaining time, and never writing them is a timeout). The parent treats
whatever artifact appears as data and validates it.
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--observation-size", type=int, required=True)
    parser.add_argument("--action-size", type=int, required=True)
    parser.add_argument("--artifact", required=True)
    opts = parser.parse_args()
    ready = os.dup(1)   # the judge's pipe; not inherited by child processes
    os.dup2(2, 1)

    # Pristine dependencies only; the heavy imports are outside the budget.
    import jax
    import jax.numpy as jp
    from jax import export
    from jax._src.export import serialization  # noqa: F401  (needs flatbuffers)
    import brax.training.agents.ppo.train  # noqa: F401
    import mujoco  # noqa: F401
    from mujoco import mjx  # noqa: F401

    os.write(ready, b"I")

    sys.path.insert(0, opts.repo)
    from frontier_solution.train import train

    os.write(ready, b"T")
    began = time.monotonic()
    policy = train(opts.seed, float(opts.budget_seconds))
    os.write(ready, b"D")
    print(f"train_entry: train() returned after {time.monotonic() - began:.1f}s", flush=True)
    if not callable(policy):
        raise TypeError("train() must return a callable policy(obs_state, rng)")

    def batched(observations, keys):
        actions = jax.vmap(policy)(observations, keys)
        return jp.asarray(actions, dtype=jp.float32)

    exported = export.export(jax.jit(batched))(
        jax.ShapeDtypeStruct((opts.envs, opts.observation_size), jp.float32),
        jax.ShapeDtypeStruct((opts.envs, 2), jp.uint32))
    if tuple(exported.out_avals[0].shape) != (opts.envs, opts.action_size):
        raise ValueError("policy must return one action vector per observation")
    body = bytes(exported.serialize())
    temporary = opts.artifact + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(body)
    os.replace(temporary, opts.artifact)
    print(f"train_entry: exported {len(body)} bytes", flush=True)
    os.write(ready, b"E")


if __name__ == "__main__":
    main()

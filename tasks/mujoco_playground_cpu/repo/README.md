# Go1 Joystick Policy Training

Deliver a training procedure that, within a fixed 720-second wall-clock budget on one NVIDIA
H100, produces a better Unitree Go1 joystick locomotion policy than the repository's own
procedure in the same budget. The policy is trained from scratch inside each verification and
evaluated in the judge's unmodified environment; higher episode reward scores higher. A
beats-reference criterion is evaluated alongside the score (see Scoring).

## Task

This repository is MuJoCo Playground ("Demonstrating MuJoCo Playground", RSS 2025) at commit
`254c38f9c240012518dbe61d1267bc375ff827e4`. A joystick policy reads proprioceptive observations
and a commanded planar velocity and yaw rate and outputs joint position targets, so that the robot
tracks the command without falling. Nothing has been removed: the Go1 environment and reward
(`mujoco_playground/_src/locomotion/go1/joystick.py`), domain randomisation (`randomize.py` beside it),
PPO configuration (`mujoco_playground/config/locomotion_params.py`) and training script
(`learning/train_jax_ppo.py`) together are the reference. The shipped `frontier_solution/train.py`
is that reference procedure (Brax PPO on `Go1JoystickFlatTerrain`, the repository's configuration
as shipped, 200 million environment steps), so the starting tree runs unchanged; it already uses the repository's full
step count, so there is no shortened step count to restore.

## What you may change

- files below `frontier_solution/` (keeping `frontier_solution/train.py`); at most 50 changed files
- `mujoco_playground/_src/locomotion/go1/joystick.py`,
  `mujoco_playground/_src/locomotion/go1/randomize.py`
- `mujoco_playground/config/locomotion_params.py`

Open: rewards, commands or a curriculum, training-time randomisation and perturbations, training
terrain, the network, PPO hyperparameters or another algorithm. Fixed: the robot model and XML,
the actor observation/action interface, the budget and the whole evaluation. The policy must be
learned in the verification run: no pretrained parameters or encoded trained policy or value
function. Added lines with over 16,384 characters in runs of 120+ from `A-Z a-z 0-9 + / = _ -`,
or over 20,000 numeric literals, are rejected as `content_policy`.

## Interface

```python
# frontier_solution/train.py
def train(seed: int, deadline_seconds: float) -> policy: ...

# the returned object
def policy(obs_state: jax.Array, rng: jax.Array) -> jax.Array: ...
```

- `seed` is the training seed of the workload.
- `deadline_seconds` is the training budget: the wall-clock time, counted from
  the moment `train` is called, by which `train` must have returned. It is the
  same clock the judge enforces, and its value is 720.0.
- `obs_state` is the actor observation `obs["state"]` of one environment:
  `float32`, shape `(48,)`. The privileged observation is not available to the
  policy.
- `rng` is a JAX PRNG key as produced by `jax.random.PRNGKey` (`uint32`, shape
  `(2,)`). A fresh key is supplied at every control step; a deterministic policy
  may ignore it.
- The return value is the action for that environment: `float32`, shape
  `(12,)`, finite. It is passed to the environment unchanged.
- `policy` must be traceable by `jax.jit` and `jax.vmap` with exactly those
  shapes and must depend on nothing but its two arguments and values captured
  at trace time. Trained parameters are captured by closure.

After `train` returns, the judge traces `jax.vmap(policy)` for a batch of 2,048
environments and serialises it with `jax.export`. Only that serialised tensor
program leaves the training process; the evaluation never imports your Python.
A policy that cannot be exported this way (Python side effects, host callbacks,
data-dependent Python control flow) is invalid. The policy is stateless across
control steps: it receives the current observation only.

For example, the shipped reference ends with:

```python
brax_policy = make_inference_fn(params, deterministic=True)
privileged_placeholder = jp.zeros_like(params[0].mean["privileged_state"])

def policy(obs_state, rng):
  observation = {"state": obs_state, "privileged_state": privileged_placeholder}
  action, _ = brax_policy(observation, rng)
  return action
```

## Correctness

Score 0 if:
- `train` does not return within 720 s of the call. Everything inside `train` is charged
  (environments, JIT, training, your own evaluation); process start and imports are not. An
  overrun is killed; the reference needs about 445-465 s.
- Importing your modules takes over 180 s or exporting the policy over 300 s (neither may train
  or tune).
- `train` raises or returns no exportable policy with the interface above, or the exported policy
  exceeds 256 MiB (268,435,456 bytes).
- The policy gives a non-finite action or episode measurement, or one evaluation exceeds 600 s.
- Your run exceeds the suite time limit: 5,400 s for a train verification, the evaluation limit in
  `/frontier/task/README.md` for final (both include provisioning).

Reference or judge failures are verifier errors, never scores. No network during verification.

## Workload

A workload is one training seed, one evaluation condition and one evaluation seed. For each
training seed the judge calls your `train` once and the unmodified reference `train` once, on the
same H100 under the same budget; each policy is evaluated in every condition the suite pairs with
that seed.

The conditions use only what the repository provides: `flat` is `Go1JoystickFlatTerrain` with its
default configuration, `push` the same with `pert_config.enable = True`, and `rough` is
`Go1JoystickRoughTerrain` with its default configuration.

| suite | workloads |
|---|---|
| train | 2: one training seed in `flat` and `push` |
| final | 15: five further training seeds, each in `flat`, `push` and `rough` |

Seeds are private and the final seeds are disjoint from the train seeds; `rough` is never in
train feedback. One evaluation runs 2,048 parallel environments for 1,000 control steps (20 s of
simulated time) with the policy's deterministic output, the environment's own command sampling,
resampling and observation noise, and no domain randomisation or automatic reset. Training is
noisy: across seeds the reference's `episode_reward` varies with a standard deviation of about
0.2-0.3 on flat ground, and a seed does not reproduce bit-for-bit, so seeds cannot pair your run with the
reference's.

## Scoring

All metrics come from the unmodified environment and reward scales. An environment is alive at a
step until its own fall detection first terminates it; later steps count for nothing.
- `episode_reward`, `tracking_lin_vel`, `tracking_ang_vel`: the per-environment sum over alive
  steps of `state.reward`, `state.metrics["reward/tracking_lin_vel"]` and
  `state.metrics["reward/tracking_ang_vel"]`, averaged over the 2,048 environments.
- `energy`, `action_rate`: `state.metrics["reward/energy"]` and `["reward/action_rate"]` summed
  over all environments and alive steps, divided by the number of alive steps (negative
  penalties, per alive step).

Higher is better for all five.

```text
r     = max(0, mean over workloads of episode_reward)
score = 100 * r / (r + 30)
```

Anchors: mean episode reward <= 0 scores 0, 30 scores 50 (the shipped reference is close to 30),
every finite reward below 100.

**Beats the reference** when every validity check passes and, on every final workload, all five
metrics are strictly higher than those of the reference trained in the same verification (no
tolerance). It does not change the score.

## Running locally

On a GPU runner, `frontier_check.py` trains with your `frontier_solution`, exports the policy the
way the judge does and prints the five metrics per condition (in your working tree's environment
code and with public seeds):

```sh
cd /workspace/repo
/opt/frontier/task-runtime /opt/frontier/runtime-python frontier_check.py \
    --seed 0 --conditions flat,push,rough
```

## Environment

Python 3.12, JAX 0.4.35 with CUDA 12, Brax 0.12.1, MuJoCo and MJX 3.2.7, Flax 0.10.2, Optax 0.2.4
(hash-locked in `/frontier/task/runtime/requirements-gpu.lock`); one NVIDIA H100. The CPU
container has no JAX. The Unitree Go1 assets of MuJoCo Menagerie (commit
`14ceccf557cc47240202f2354d684eca58ff8de4`) are in `mujoco_playground/external_deps/`; other
robots' assets are not included. `UPSTREAM_README.md` is the repository's original README.

## Notes

A train verification trains your procedure and the reference once each, so it takes about 20 to
40 minutes.

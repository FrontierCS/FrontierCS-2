# Go1 Joystick Policy Training

## Problem

This repository is MuJoCo Playground (Zakka et al., arXiv:2502.08844) at commit
`254c38f9c240012518dbe61d1267bc375ff827e4`. The paper trains a joystick
locomotion policy for the Unitree Go1 quadruped in minutes on a single GPU: a
policy that reads proprioceptive observations and a commanded planar velocity
and yaw rate, and outputs joint position targets so that the robot tracks the
command without falling.

Your task is to deliver a training procedure that, within a fixed wall-clock
budget on one NVIDIA H100, produces a better Go1 joystick policy than the
repository's own procedure does in the same budget. Nothing has been removed
from the repository. Its Go1 environment and reward
(`mujoco_playground/_src/locomotion/go1/joystick.py`), domain randomisation
(`.../go1/randomize.py`), PPO configuration
(`mujoco_playground/config/locomotion_params.py`) and training script
(`learning/train_jax_ppo.py`) are all present, and together they are the
reference you are measured against. The policy is trained from scratch inside
each verification; it is never trained ahead of time.

## Required behavior

`frontier_solution/train.py` must expose `train(seed, deadline_seconds)` and
return a policy function as specified under Interface. The shipped
`frontier_solution/train.py` already does this: it is the reference procedure
(Brax PPO on `Go1JoystickFlatTerrain` with the repository's PPO configuration
exactly as shipped: 200 million environment steps), so the starting tree runs
unchanged. The reference is the artifact's own configuration, not a shortened
one, so raising the step count back to the repository's is not an available
improvement.

The design space is open inside the editable paths: reward terms and their
structure, the command distribution or a command curriculum, training-time
domain randomisation and perturbations, terrain used for training, the network
architecture, PPO hyperparameters, or a different learning algorithm.

Fixed, and outside the editable paths: the robot model and physics XML, the
actor observation and action interface, the wall-clock budget, and the whole
evaluation (environments, command sampling, reward terms and scales, random
streams, metrics).

## Workload

A workload is one training seed, one evaluation condition and one evaluation
seed. For each training seed the judge calls your `train` once, and separately
calls the unmodified reference `train` once, each on the same H100 and each
under the same wall-clock budget. Each resulting policy is then evaluated in
every condition that the suite pairs with that training seed.

Evaluation conditions use only what the repository provides:

| condition | environment | configuration |
| --- | --- | --- |
| `flat` | `Go1JoystickFlatTerrain` | default |
| `push` | `Go1JoystickFlatTerrain` | default with `pert_config.enable = True` |
| `rough` | `Go1JoystickRoughTerrain` | default |

One evaluation runs 2,048 parallel environments for 1,000 control steps (20 s
of simulated time each) with the policy's deterministic output, the
environment's own command sampling and resampling, the environment's own
observation noise, and no domain randomisation and no automatic reset.

The development suite has 2 workloads: one training seed evaluated in `flat`
and `push`. The final suite has 15 workloads: five further training seeds, each
evaluated in `flat`, `push` and `rough`. The seeds are private, and the final
seeds are disjoint from the development seeds. Rough terrain is never part of
development feedback.

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

## Public GPU checks

Your CPU container has no JAX runtime. Upload your working repository to the
Agent GPU runner as described in `/frontier/task/README.md`; its image has the
same Python 3.12 environment the verifier uses (JAX 0.4.35 with CUDA 12, Brax
0.12.1, MuJoCo and MJX 3.2.7, Flax 0.10.2, Optax 0.2.4), listed with hashes in
`/frontier/task/runtime/requirements-gpu.lock`. The Unitree Go1 assets of
MuJoCo Menagerie (commit `14ceccf557cc47240202f2354d684eca58ff8de4`) are
already in the tree under `mujoco_playground/external_deps/`, so importing
`mujoco_playground` does not need the network; assets of other robots are not
included.

`frontier_check.py` trains with your `frontier_solution`, exports the policy
the way the judge does and prints the five metrics per condition:

```sh
cd /workspace/repo
/opt/frontier/task-runtime /opt/frontier/runtime-python frontier_check.py \
    --seed 0 --conditions flat,push,rough
```

It measures in your working tree's environment code and with public seeds; the
judge measures in its own unmodified copy with private seeds. `UPSTREAM_README.md`
is the repository's original README.

## Validity contract

A submission is invalid, and scores zero, if any of the following holds.

- The patch touches a path outside the editable surface, is empty, contains
  binary content, or exceeds 262,144 bytes.
- A training run does not finish inside the wall-clock budget. The budget is
  **720 seconds per training run**: wall-clock time on the verifier's H100 from
  the moment the judge calls your `train` until `train` returns. Everything
  inside `train` is charged, identically for your procedure and for the
  reference: building environments, JIT compilation, training, and any
  evaluation you run. Starting the process, importing JAX, Brax and MuJoCo, and
  importing `frontier_solution` and `mujoco_playground` happen before the clock
  starts and are not charged. A run that overruns is killed. Leave margin for
  machine variance rather than planning to return in the last seconds; the
  reference procedure needs roughly 445 to 465 seconds.
- Work is moved outside that clock. Importing your modules must finish within
  180 seconds and exporting the returned policy within 300 seconds; both are
  generous limits for defining code and tracing a network, and neither may be
  used to train or to tune the policy.
- Training raises, returns something that is not a policy with the interface
  above, or returns a policy that cannot be exported.
- The exported policy produces a non-finite action or non-finite episode
  measurements in any evaluation, or one evaluation does not finish within 600
  seconds (the reference policy's evaluation takes one to two minutes including JIT).
- The policy is not learned inside the verifier run. Shipping pretrained
  parameters, or any encoding of a previously trained policy or value function,
  in the patch is not allowed. Hyperparameters, reward weights, architecture
  definitions and curricula are of course allowed. As a backstop, a patch is
  rejected with "submission embeds parameter-like data; the policy must be
  learned inside the verifier run" if, over its added lines, runs of 120 or
  more consecutive characters from `A-Z a-z 0-9 + / = _ -` total more than
  16,384 characters, or there are more than 20,000 numeric literals. These
  thresholds are far above what source code needs; staying under them does not
  make shipped parameters allowed.

The evaluation environment is never built from submitted code. Training runs in
its own process from your patched tree; the evaluation runs in a separate
process that imports the environment only from the judge's unmodified copy of
this repository and receives your policy as serialised data. The network is
unavailable during verification. A failure of the reference side, or of the
judge's own machinery, is reported as a verifier error and is never turned into
a score for your submission.

## Scoring

Every measurement below is taken in the unmodified environment with the
unmodified reward scales, from quantities the environment itself reports. In
one evaluation, an environment is *alive* at a control step if it has not
terminated at or before that step (termination is the environment's own fall
detection). Steps after an environment's first termination contribute nothing.

Episode metrics, each the mean over the 2,048 environments of a per-environment
sum over that environment's alive steps:

- `episode_reward`: sum of the environment's total step reward `state.reward`.
- `tracking_lin_vel`: sum of `state.metrics["reward/tracking_lin_vel"]`.
- `tracking_ang_vel`: sum of `state.metrics["reward/tracking_ang_vel"]`.

Per-step metrics, each the sum over all environments and all alive steps
divided by the total number of alive steps:

- `energy`: mean of `state.metrics["reward/energy"]` per alive step.
- `action_rate`: mean of `state.metrics["reward/action_rate"]` per alive step.

The two per-step quantities are already negatively scaled by the environment
(they are penalties), so for all five metrics **higher is better**. Falling
early lowers the three episode metrics. The two penalties are normalised per
alive step so that a policy is neither rewarded for falling early nor penalised
for surviving longer.

The score to maximise is

```text
r = max(0, mean_workloads(episode_reward))
score = 100 * r / (r + 30)
```

Workloads have equal weight. The score is strictly increasing in `r`: a mean
episode reward of zero or less scores `0`, a mean episode reward of 30 scores
`50`, and every finite reward scores below `100`, with no finite upper
clipping, so better policies remain strictly ordered as the score approaches
`100`. The half-saturation reward `30` is a fixed challenge constant, not a
measurement of any implementation, and the score does not depend on the
reference's measurements. For orientation only, not as a target: the shipped
reference procedure's mean episode reward is close to 30, so it scores near 50.
The `score_unbounded` output carries the same naturally bounded value; there is
no separate clipped score. Invalid submissions receive zero.

## Calibration status

The reference procedure has been run repeatedly on single H100s outside this
task's verifier image, and through this task's evaluator in an image built
from its verifier recipe; qualification through the platform's Driver with the
published images is pending.
Reinforcement-learning training is noisy: across training seeds the reference's
`episode_reward` varies with a standard deviation of roughly 0.2 to 0.3 on
flat ground, and the same training seed run twice on a GPU does not reproduce
bit-for-bit. Seeds are therefore samples of noise, not a way to pair your run
with the reference's run. Within one training pair the comparisons are strongly but
not totally correlated: a run that is better on one metric tends to be better
on the others, in every condition. No tolerance is applied to any comparison. No
submission that meets the stricter criterion below is known to exist. The
reference procedure has been run against itself through this task's evaluator
on an H100 (one development suite and one final suite); it did not meet the
stricter criterion against itself.

## Feedback and grading

The configured limit is 3 development submissions. A development verification
trains one seed per side and returns, for your policy only,
`development_mean_episode_reward`, `development_mean_tracking_lin_vel`,
`development_mean_tracking_ang_vel`, `development_mean_energy` and
`development_mean_action_rate`: the arithmetic mean of each metric over the two
development workloads, in the units defined under Scoring. It also returns
`beat_spec_progress`: for each of the five metrics, the number of development
workloads on which your value was strictly better than the reference's, as
`passed`, `failed`, `total`, `pass_percent` and `fail_percent`. It does not
identify workloads and does not return any reference value or margin.

Because training is noisy, a development result is one sample. Two development
submissions of the same patch will not return the same numbers.

In addition to the continuous score, a separate stricter beats-reference
criterion compares your policy with the reference policy trained in the same
verification, under the same budget, on the same hardware, on every graded
workload. It is met only if all validity checks pass and, on every workload,
your `episode_reward`, `tracking_lin_vel`, `tracking_ang_vel`, `energy` and
`action_rate` are each strictly higher than the reference's, with no tolerance.
Reference values, comparison margins, private workload identities,
per-workload rows and all final measurements are not disclosed. The continuous
score remains the optimisation objective whether or not this stricter
criterion is met.

## Resources

The control pod has 3 vCPUs, 12,288 MiB RAM and 10,240 MiB workspace storage.
Verification uses one Driver-provisioned NVIDIA H100. A development
verification has a 5,400-second GPU limit and the independent final
verification a 14,400-second limit (a typical final is ten trainings of
470 to 720 seconds and 30 evaluations of under a minute); both include provisioning the runner,
pulling its image and uploading inputs. The verifier GPU pool is 30,600
seconds (three development verifications and one final). Your own GPU
debugging budget is one H100 runner at a time and 14,400 seconds in total.
That budget is spent from the moment a runner is allocated: a fresh runner
first pulls a multi-gigabyte image, and the first JAX/CUDA initialisation and
each JIT compilation take tens of seconds to minutes, so it is not all
training time.

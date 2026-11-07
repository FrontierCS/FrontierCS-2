# Closed-loop action sampling

Implement the conditional action sampler of a frozen diffusion-policy controller for the Push-T
robot task, running on CPU. Higher is better: the score rewards task reward above that of a fixed
20-step DDPM sampler, at lower prediction latency and episode wall time. A valid submission changes
only the body of `conditional_sample` (plus optional helper modules), follows the sampler interface
below, and runs every episode within its time limits.

## Task

A simulated robot pushes a T-shaped block into a target region. The controller is the Diffusion
Policy transformer for low-dimensional keypoint observations (Chi et al., "Diffusion Policy",
RSS 2023), loaded from a frozen EMA checkpoint with its normalizer, a 2-step observation window and
receding-horizon execution of 8 actions per prediction. The original sampler runs 100 DDPM
denoising steps; in this tree its body is removed (`NotImplementedError`).

You write the reverse sampling that turns observations into a normalized action sequence using the
frozen transformer. The number and placement of denoising steps, stochastic or deterministic
updates, and the loop implementation are yours; the result does not have to match the original
sampler's tensor or step count. The controller architecture, weights, training, environment and
reward are fixed.

## What you may change

- The body of `DiffusionTransformerLowdimPolicy.conditional_sample` in
  `diffusion_policy/policy/diffusion_transformer_lowdim_policy.py`. Everything else in that file,
  including the signature and all whitespace outside the body, stays byte-identical; the file is at
  most 131,072 bytes.
- New helper modules under `diffusion_policy/sampler_extensions/`: `.py` files, valid Python, not
  executable, no path part starting with `.`; at most 16 files and 65,536 bytes in total.
- No other file may be changed or deleted, permissions stay as they are, and only regular files
  (no symlinks or hard links) are allowed.

A change outside these rules is rejected as `out_of_scope` (edits or new files elsewhere, non-UTF-8
source), `forbidden_file_type` (links, executable helpers, permission changes), `too_large` (the
byte or file limits) or `content_policy` (a helper that does not parse).

## Interface

```python
def conditional_sample(self,
        condition_data, condition_mask,
        cond=None, generator=None,
        # keyword arguments to scheduler.step
        **kwargs
        ):
```

The judge loads your policy file in a separate sampler process, one per episode, on CPU with one
thread. There `self` is rebuilt from the checkpoint's public configuration:

- `self.model(sample, timestep, cond)` evaluates the frozen transformer in the judge process.
  `sample` is float32 with the shape of `condition_data` (batch 1), `cond` is the `cond` you
  received, and `timestep` is a scalar or one-element tensor with values in `[0, 100)`. It returns
  a float32 tensor shaped like `sample`.
- `self.noise_scheduler` is a fresh diffusers `DDPMScheduler` built from the checkpoint's scheduler
  configuration (`num_train_timesteps = 100`), and `self.num_inference_steps` is 100. `horizon` is 10,
  `n_obs_steps` 2, `n_action_steps` 8 and `obs_as_cond` true; `obs_dim` and `action_dim` are the
  checkpoint's.
- `torch.randn` and `torch.randn_like` draw from the judge's per-episode generator, seeded from the
  episode's diffusion seed. Call them on CPU without a `generator`, with at most 65,536 elements per
  call. The `generator` argument is always `None` and `kwargs` is empty.

Inputs: `condition_data` float32 `(1, 10, action_dim)`, `condition_mask` bool of the same shape,
`cond` float32 `(1, 2, obs_dim)` (normalized observations). Return a CPU tensor with the shape and
dtype of `condition_data`, all entries finite, and every entry where `condition_mask` is true equal
to `condition_data`; the caller unnormalizes it and slices the actions. With data `[1.0, 2.0, 3.0]`
and mask `[True, False, True]`, `[1.0, 0.5, 3.0]` is valid and `[0.0, 0.5, 3.0]` is not. With this
checkpoint `condition_mask` is all false.

Each call must invoke `self.model` at least once and at most 200 times. An episode may make at most
38,000 random calls.

## Correctness

During measurement, any of these rejects the submission as `content_policy` (score 0):
- the sampler process raises or exits, or writes anything to stdout (the protocol channel);
- a returned tensor with the wrong shape or dtype, a non-finite entry, or a changed masked entry;
- a prediction with no `self.model` call or more than 200; more than 38,000 random calls in an
  episode;
- a model or random request outside the interface above (dtype, shape, timestep, device).

Any of these scores the submission 0 (not a rejection):
- a sampler exchange is still pending 130 s after your sampler process started;
- one of your episode processes exceeds 150 s, including imports, checkpoint loading and cleanup;
- the suite's measurement budget runs out: train runs its 12 episode processes in 1,860 s, final
  its 24 within the evaluation timeout stated in `/frontier/task/README.md` (reference episodes
  included), and an episode starts only if at least 155 s remain.

## Workload

One workload is one Push-T initial condition (environment seed) run as three complete episodes,
each with its own diffusion seed. An episode uses the unchanged environment runner with one
environment, a 300-step cap and observation/prediction/action horizons 2/10/8, so it makes at most
38 predictions. It ends on the environment's original success or at the cap (not a success).

| suite | workloads | episodes per side |
|---|---|---|
| train | 2 initial conditions | 6 |
| final | the 2 train initial conditions with new diffusion seeds, plus 2 held-out | 12 |

- All initial-condition and diffusion seeds are distinct, final diffusion seeds are disjoint from
  train, and seeds and workload identifiers are withheld.
- Each repetition runs your sampler and the reference with the same seeds, alternating which goes
  first; each episode is a fresh process.
- Timed: each `predict_action` call (normalization, sampling, protocol round-trips) and each
  whole episode process (startup, checkpoint loading, sampler start, episode, cleanup).

## Scoring

Per workload `w`: `Q` is the mean over its three episodes of the episode-maximum reward (0 to 1),
`L` the mean `predict_action` seconds over all its predictions, and `T` the median episode-process
seconds. `Qn`, `Ln`, `Tn` are fixed, pre-measured values of the original sampler run with 20 DDPM
steps on the same workload (`Qn < 1`).

```text
G(w)  = max(0, (Q - Qn) / (1 - Qn))
u(w)  = G(w) * min(Ln / L, Tn / T)
U     = mean of u(w) over the workloads (2 in train, 4 in final)
score = 100 * U / (1 + U)          score_unbounded = U
```

Anchors: reward at or below `Qn` gives `u = 0`; reward 1 at the naive latency and wall time gives
`u = 1`, and `U = 1` scores 50. The untouched tree raises
`NotImplementedError` and is rejected. Episode success rate is not in the score.

**Beats the reference** when, on every workload of the suite (2 train, 4 final), all four hold
against the original 100-step DDPM sampler:
mean episode-maximum reward `Q` at least the reference's; episode success rate (fraction of the
three episodes reaching original success) at least the reference's; mean `predict_action` seconds
`L` strictly lower; median episode wall time `T` strictly lower. There is no tolerance. This
criterion does not change the score.

## Running locally

The Agent machine has the judge's Python runtime as `python3`, but not the checkpoint or the
evaluator. Test `conditional_sample` by constructing the policy with a stand-in `model` and a
`DDPMScheduler`.

## Environment

- CPU only: 2 CPUs, one Torch/BLAS thread, a 4 GiB address-space limit per episode process
  (its sampler process inherits the same limit).
- Python 3.10.20, torch 2.2.2+cpu, numpy 1.23.5, diffusers 0.11.1.
- The official Push-T transformer low-dimensional checkpoint (EMA weights).

## Notes

The workloads are representative manipulation episodes, not initial states to recognize: do not
special-case initial conditions, seeds or the train/final phase. Also not permitted: reading
environment, judge or private state; reseeding random generators; background work that hides cost
outside a prediction; changing retained code at run time. Submissions are reviewed for these.

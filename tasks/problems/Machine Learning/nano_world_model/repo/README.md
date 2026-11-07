# Future Video Sampling

Implement the latent-video sampler of the NanoWM world model so that its predicted future frames
match the recorded ones as closely as possible. The score rises with PSNR and SSIM and falls with
LPIPS, relative to a hidden reference sampler measured in the same verification (the reference
scores 50). A valid submission returns a finite latent tensor of the required shape for every
clip and passes the source policy below.

## Task

NanoWM (paper: *Nano World Models*) predicts future video frames from a context frame and a
sequence of actions. This tree runs NanoWM's evaluation pipeline with a fixed NanoWM-B/2 model,
an SD-VAE codec (a latent is the encoder's representation of an RGB frame) and recorded DINO-WM
Point Maze trajectories (a 2-D maze with two-component actions). The latent-video sampler, its
schedule utilities, the ordinary DDIM transition, and the Gaussian and Flow Matching sampling
hooks are removed (`NotImplementedError`); supply them. The end-to-end evaluation enters through
`dfot_sample` and uses the Gaussian backend; the Flow Matching hooks must stay importable but are
not run or scored.

## What you may change

- `src/diffusion/df_sample.py`, `src/diffusion/gaussian_diffusion.py`,
  `src/diffusion/flow_matching.py`
- new Python helpers under `src/diffusion/frontier_solution/` (`.py` or `.pyi`; every directory
  name and filename stem a valid Python identifier; `.pyi` files are not executable helpers).

Keep the public names and signatures listed under Interface. Everything else is fixed: model
weights and architecture, action conditioning, encoding and decoding, clip selection, the writer,
the metrics and the evaluator. No file may be deleted.

Source policy on the editable files (violation: `content_policy`):
- absolute imports only from `diffusion`, `collections`, `dataclasses`, `enum`, `functools`,
  `itertools`, `math`, `numpy`, `operator`, `torch`, `tqdm`, `typing`;
- the source text, case-insensitive, must not contain `judge_data`, `frontier_submission_role`,
  `frontier_model_root`, `validation_fixed_subset`, `torch.load`, `numpy.load`, `np.load`,
  `getenv` or `environ`;
- the names `__builtins__`, `__import__`, `builtins`, `compile`, `eval`, `exec`, `getattr`,
  `globals`, `input`, `locals`, `open`, `setattr`, `vars` are forbidden, as are the attributes
  `_getframe`, `ag_frame`, `cr_frame`, `f_back`, `f_code`, `f_globals`, `f_locals`, `fromfile`,
  `genfromtxt`, `gi_frame`, `load`, `loadtxt`, `memmap`, `save`, `tb_frame`, `with_traceback` and
  every dunder attribute access;
- every editable file must parse and compile.

## Interface

Every judged clip makes this call (`context` and `action` are float32 CUDA tensors):

```python
sample = dfot_sample(
    diffusion=diffusion, model=model, shape=(1, 4, 4, 32, 32),
    context=context,                 # (1, 1, 4, 32, 32): the encoded first frame
    n_context_frames=1, scheduling_mode="sequential", num_sampling_steps=250,
    model_kwargs={"y": None, "action": action},   # action (1, 4, 10)
    device=torch.device("cuda:0"), progress=False, eta=0.0, clip_denoised=False,
    n_generate_frames=None, history_stabilization_level=0.02,
)
```

Return a `torch.Tensor` of shape `(1, 4, 4, 32, 32)` (batch, frame, channel, height, width),
float32, finite, frames in chronological order, all four frames including the
context position (which is not measured). Latents use the codec's scaling factor, like `context`.
All clips are sampled in one process, one call per clip, on a GPU machine of their own that holds
only the submitted tree, the model, the codec, and per clip the context frame and actions; the
global RNG state carries over from call to call. The model is in eval mode with gradients
disabled; `model(x, t, y=None, action=action)` returns the
shape of `x`, where `t` holds diffusion indices per batch or per batch and frame, and
`action[:, j]` drives the transition from frame `j` to `j+1`. It predicts
`v = sqrt(alpha_bar[t]) * noise - sqrt(1 - alpha_bar[t]) * clean_latent`
(`diffusion.alphas_cumprod` is `alpha_bar`). The diffusion object is fixed: 1,000 timesteps,
`noise_schedule="squaredcos_cap_v2"`, `pred_name="v"`, `zero_terminal_snr=True`,
`snr_gamma=5.0`.

The removed callables:
- `generate_full_sequence_schedule`, `generate_pyramid_schedule`,
  `generate_sequential_schedule(sampling_timesteps, horizon) -> np.ndarray`: the schedule of
  sampling indices for `horizon` generated frames;
- `ddim_idx_to_timestep(indices, num_timesteps, sampling_timesteps) -> Tensor`, shape-preserving;
- `generate_scheduling_matrix(num_frames, num_timesteps, sampling_timesteps, mode="pyramid",
  n_context_frames=0) -> Tensor`: one column per frame including context; `mode` is
  `full_sequence`, `pyramid` or `sequential`;
- `dfot_sample(...) -> Tensor` (defaults `scheduling_mode="pyramid"`, `num_sampling_steps=50`,
  `progress=True`, `history_stabilization_level=0.0`), returns exactly `shape`;
- `GaussianDiffusion.ddim_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None,
  model_kwargs=None, eta=0.0)`, `GaussianDiffusion.dfot_ddim_sample(model, x, curr_t, next_t,
  clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0)` and the `FlowMatching`
  `dfot_ddim_sample`: each returns `{"sample", "pred_xstart"}`, both shaped like `x`;
- `GaussianDiffusion.dfot_sample_loop` and `FlowMatching.dfot_sample_loop(model, shape,
  scheduling_matrix, context=None, n_context_frames=0, clip_denoised=True, denoised_fn=None,
  model_kwargs=None, device=None, progress=False, eta=0.0) -> Tensor` of `shape`.

Only `context` and `action` reach the sampler (not the other frames, targets, states or names).
The sampler writes no files. The caller decodes, clamps RGB to `[-1, 1]`, rescales to `[0, 1]` and
writes `pred` and `gt` float32 `[3, 4, 256, 256]` (C, T, H, W) per clip; scoring uses frames 1–3.
Example: a decoded `[-1.0, 0.0, 1.0]` pixel is written as `[0.0, 0.5, 1.0]`.

## Correctness

The score is 0 when:
- the patch is rejected or a file fails the source policy (no measurement);
- after the trusted reference run, the sampler raises, times out, runs out of memory or loses its
  machine's session, or returns a wrong type, shape or dtype, non-finite values, or output for
  fewer clips than the suite (then every clip also fails the comparison).

Sampling all clips has 4,950 seconds, counted from when the sampling machine is ready. It runs
the retained `src/main.py` (dispatch, checkpoint loader, Point Maze frames, codec, sampler,
writer); the judge decodes the returned latents with the same pipeline and measures them on its
own machine. A failure of the reference run or the runtime is an unscored failed verification.

## Workload

One workload is one clip: a 20-step slice of a recorded Point Maze episode, with images at offsets
0, 5, 10, 15 (one context frame, three scored future frames) and actions at offsets 0–19.

| suite | clips |
|---|---|
| train (development) | 4 |
| final | 6 |

The suites are disjoint (different episodes) and drawn from the same Point Maze source and
distribution; membership is withheld.
Images are resized to 256×256 (bilinear, `align_corners=False`) and mapped to `[-1, 1]`. Each
action component is normalized as `(a - mean) / (std + 1e-6)` over all 200,000 action records,
and each group of five two-component actions is packed into ten values. Evaluation uses batch
size 1, seed 3407, float32 (matmul precision `high`, inherited TF32 settings), no compilation, one
sampling call per clip. Speed is not scored.

## Scoring

Per clip, `P_i`, `S_i`, `L_i` are the means over the three future frames of PSNR (dB; `[0,1]` RGB,
range 1.0, epsilon 1e-8), SSIM (11×11 window, sigma 1.5, 3 channels) and VGG-LPIPS (inputs mapped
to `[-1,1]`). `P`, `S`, `L` are their means over all clips; `P*`, `S*`, `L*` the reference's on the
same clips:

```text
g_P = (P - P*) / 2.5
g_S = log2((1 - S*) / (1 - S)) / 0.3        # 1 - S, 1 - S*, L, L* floored at 1e-6
g_L = log2(L* / L) / 0.4
G   = min(1, max(0, (min(g_P, g_S, g_L) + 0.06) / 0.05))
t_m = tanh(g_m)
score = 50 + (50/3) * (sum of G*t_m over positive t_m + sum of t_m over negative t_m)
```

Anchors: the starter scores 0; matching the reference scores 50; the score stays strictly between
0 and 100 (+1 dB mean PSNR alone ≈ 56.33; +1 dB PSNR, 10% lower `1 - SSIM` and 10% lower LPIPS together ≈ 70.17). Gains count only while every
`g_m > -0.06` (fully while `≥ -0.01`), so one metric cannot pay for another.

**Beats the reference** requires, on every final clip, PSNR and SSIM strictly higher and LPIPS
strictly lower than the reference's (no tolerance; ties fail). It does not change the score.

## Running locally

On a GPU runner (prefix commands with `/opt/frontier/task-runtime`, see
`/frontier/task/README.md`), `bash frontier-build.sh` compiles `src/` and imports the entrypoint
and `dfot_sample`. `bash frontier-bench.sh` runs the retained `src/main.py` evaluation but needs
`NANOWM_CHECKPOINT` (convert `/opt/frontier/models/point-maze/model.safetensors` with `torch.save`),
`NANOWM_SUBSET`, `DATASET_DIR` and `VAE_MODEL_PATH=/opt/frontier/models/sd-vae`; the runner has no
Point Maze data, so without your own data only the build check and direct `dfot_sample` calls with
the model on synthetic tensors can run.

## Environment

Python 3.11, PyTorch 2.4.1 (CUDA 12.4), torchvision 0.19.1, NumPy 1.26.4, diffusers 0.30.3, PIQA
1.3.2, LPIPS 0.1.4; the Point Maze checkpoint, SD-VAE and VGG weights are offline. Verification
runs on one NVIDIA A100 80GB.

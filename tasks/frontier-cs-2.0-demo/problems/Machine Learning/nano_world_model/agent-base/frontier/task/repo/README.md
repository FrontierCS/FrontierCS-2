# Future Video Sampling

## Problem
NanoWM is an open-source research system for video world models: learned simulators that predict what an environment will look like after a sequence of actions. Such predictions support video generation and planning, where an agent needs to anticipate the consequences of its actions. The supplied code comes from NanoWM's evaluation pipeline and uses a fixed NanoWM-B/2 neural network with an SD-VAE image encoder and decoder; a latent is the encoder's numerical representation of an RGB frame. This task uses recorded DINO-WM Point Maze trajectories, consisting of rendered views of a simulated two-dimensional maze, two-component control actions, and simulator states. The bundled subset is about 67 MiB, with action and state metadata for 2,000 trajectories and four stored 100-frame episodes; the development suite selects four four-frame clips and the graded suite six. For each clip, the sampler receives the encoded first frame and actions and must return a finite float32 latent tensor of shape `(1, 4, 4, 32, 32)` on the supplied CUDA device, in batch, frame, channel, height, width order. The retained decoder and writer produce an RGB video, and only its three future frames, at indices 1, 2, and 3, are compared with the corresponding recorded frames. Correctness requires the specified tensor and complete output records; prediction quality is measured continuously rather than by exact pixel equality. The objective is to maximize the mean quality score derived from PSNR, SSIM, and LPIPS over the suite's two original clips; the clips added by beat spec v3 enter only the per-clip comparison.

The latent-video sampler, its schedule utilities, the ordinary DDIM transition, and the Gaussian and Flow Matching sampling hooks have been removed and must be supplied.

## Design space
The scope is the numerical implementation of the removed latent-video sampling interfaces and their Python helpers. The editable files contain schedule construction, timestep conversion, sampling loops, and sampling transitions. Their public names and call signatures belong to the retained program's interface. The end-to-end Point Maze evaluation enters through `dfot_sample` and instantiates the Gaussian backend with the supplied pretrained model. The Flow Matching hooks remain part of the import-compatible source surface, but this workload does not instantiate that backend or separately score individual schedule tables.

The sampler is given a model in evaluation mode, with gradients disabled, and a diffusion object constructed from fixed settings. Model weights and architecture, the model's action-conditioning interface, image encoding and decoding, clip selection, output writing, metric computation, and evaluator orchestration are outside the editable surface. No model training, simulator control, or choice of target observations is part of this task. Each clip's sampler call runs in a fresh process; the caller's random-number state is supplied and restored across that boundary.

## Workload
The data is a subset of the released DINO-WM Point Maze recordings. It contains 2,000 trajectories of metadata, each with 100 two-component actions and 100 four-component states, and the four 100-frame RGB episodes needed by the suites: 200,000 action records, 200,000 state records, and 400 stored images. The files and two suite manifests occupy 69,835,769 bytes, about 66.6 MiB. These are recorded simulator trajectories; the evaluator generates no synthetic cases.

The data directory has this layout. Episode identifiers and manifest membership are withheld; `<id>` denotes the zero-padded trajectory number, not a file the solver must create.

```text
point_maze/
  actions.pth              float64 Tensor[2000, 100, 2]
  states.pth               float64 Tensor[2000, 100, 4]
  seq_lengths.pth          int64 Tensor[2000], every entry is 100
  obses/episode_<id>.pth   uint8 Tensor[100, 224, 224, 3], four files
```

The first two metadata axes are trajectory and environment step. Image axes are environment step, height, width, and RGB channel. A state contains two position coordinates followed by two velocity components. The loader converts actions and states to float32, samples four images five environment steps apart, resizes images to 256×256 with bilinear interpolation and `align_corners=False`, and maps pixels to `[-1, 1]`. It normalizes each action component as `(a - mean) / (std + 1e-6)`, using the mean and sample standard deviation of all 200,000 action records, then packs each consecutive group of five two-component actions into ten values. States are not normalized and are not passed to the sampler.

The development suite contains four fixed, distinct clips and the graded suite six (two original clips each, plus two and four clips added by beat spec v3 from the same suite episodes, without overlapping frames). A clip takes a 20-step source slice, with images at offsets 0, 5, 10, and 15 and actions at offsets 0 through 19. It becomes one context frame followed by three scored future frames. The two suites use the same Point Maze source, frame shape, four-frame horizon, action schema, NanoWM-B/2 checkpoint, and SD-VAE. The fixed diffusion configuration has 1,000 timesteps, `noise_schedule="squaredcos_cap_v2"`, `pred_name="v"`, `zero_terminal_snr=True`, and `snr_gamma=5.0`. The sampling call specifies `scheduling_mode="sequential"`, 250 sampling steps, and history-stabilization level `0.02`.

Evaluation uses batch size 1, seed 3407, float32 model and VAE tensors, no mixed precision, no model compilation, and no data-loader workers. The retained entrypoint sets float32 matrix-multiplication precision to `high`; the isolated sampler inherits its TF32 backend settings. There is one sampling call per clip and no repeated quality measurement within an evaluation call. These are representative held-out trajectory slices, not records to recognize. The current development and graded suites have the same shape and horizon; graded membership differs as described under Feedback and grading. Other datasets, long-horizon rollouts, planning success, training loss, FID, FVD, and execution speed do not contribute to the score.

## Interface
The loader's unbatched record is a dictionary. Below is an actual record with large arrays abbreviated and identity fields redacted because suite membership is withheld. Decimal values are rounded only for display. `action[0]` contains the first five normalized two-component actions in temporal order, and `state[0]` is the four-component simulator state at the first image.

```text
{
  "video": float32 Tensor[4, 3, 256, 256] in [-1, 1],
           # four source frames at offsets 0, 5, 10, and 15 in the clip
           # video[0, :, 0, 0] = [-0.803922, -0.600000, -0.403922]
  "action": float32 Tensor[4, 10],
            # row 0, the clip's first five normalized environment actions:
            [ 1.37380, -0.56587,  1.04241,  1.15238, -1.44813,
              0.53725,  1.41823, -1.51165, -0.76932, -0.75166 ],
  "state": float32 Tensor[4, 4],
           # row 0, position then velocity:
           [0.867437, 3.132803, 2.618712, -0.069292],
  "video_name": "<withheld-clip-tag>",
  "meta_info": {"traj_idx": <withheld>, "start_idx": <withheld>}
}
```

For example, the first raw action in this record is `[0.792641, -0.326240]`; its loaded value is `[1.37380, -0.56587]`. The stored RGB pixel at the first frame's top-left corner is `[25, 51, 76]`, which gives the `video[0, :, 0, 0]` values above. Batching adds a leading dimension to the tensors.

The trusted caller encodes the video with SD-VAE. Only the context latent of shape `(1, 1, 4, 32, 32)` and the action tensor of shape `(1, 4, 10)` cross into the sampler process. The remaining RGB frames, target latents, simulator states, names, and trajectory metadata do not. The context latent uses the codec's scaling factor, and the decoder expects the returned latents in that same representation.

Every judged clip makes this call; `context` and `action` are float32 CUDA tensors with the shapes just given:

```python
sample = dfot_sample(
    diffusion=diffusion,
    model=model,
    shape=(1, 4, 4, 32, 32),
    context=context,
    n_context_frames=1,
    scheduling_mode="sequential",
    num_sampling_steps=250,
    model_kwargs={"y": None, "action": action},
    device=torch.device("cuda:0"),
    progress=False,
    eta=0.0,
    clip_denoised=False,
    n_generate_frames=None,
    history_stabilization_level=0.02,
)
```

The result must be a `torch.Tensor`, with shape `(1, 4, 4, 32, 32)`, dtype `torch.float32`, device `cuda:0`, and no NaN or infinity. Its frame axis is in the same chronological order as the input clip. All four latent frames are returned, including the context position; there is no separate output for metrics or targets. The context position is excluded from quality measurement.

The supplied model accepts `model(x, t, y=None, action=action)`. Here `x` has batch/frame/channel/height/width axes, `t` contains diffusion indices per batch or per batch and frame, and the returned tensor has the same shape as `x`. The fixed model associates `action[:, j]` with the transition from frame `j` to frame `j+1`. This checkpoint predicts the diffusion quantity `v`, defined by `v = sqrt(alpha_bar[t]) * noise - sqrt(1 - alpha_bar[t]) * clean_latent`; the diffusion object's `alphas_cumprod` holds `alpha_bar`. The sampler's own return value is the completed latent video.

The removed callables and their contracts are:

- `generate_full_sequence_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray` returns the `full_sequence` schedule of sampling indices for `horizon` generated frames.
- `generate_pyramid_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray` returns the `pyramid` schedule of sampling indices for `horizon` generated frames.
- `generate_sequential_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray` returns the `sequential` schedule of sampling indices for `horizon` generated frames.
- `ddim_idx_to_timestep(indices: Tensor, num_timesteps: int, sampling_timesteps: int) -> Tensor` maps schedule indices to diffusion timestep indices, preserving the input shape; `num_timesteps` names the diffusion process length and `sampling_timesteps` the sampling-index scale.
- `generate_scheduling_matrix(num_frames: int, num_timesteps: int, sampling_timesteps: int, mode="pyramid", n_context_frames: int=0) -> Tensor` returns the timestep matrix consumed by the backend sampling loop, with columns corresponding to the requested frames, including context; `mode` is one of `full_sequence`, `pyramid`, or `sequential`.
- `dfot_sample(diffusion, model, shape, context=None, n_context_frames=0, scheduling_mode="pyramid", num_sampling_steps=50, model_kwargs=None, device=None, progress=True, eta=0.0, clip_denoised=False, n_generate_frames=None, history_stabilization_level=0.0) -> Tensor` is the public latent-video sampler and returns exactly `shape`.
- `GaussianDiffusion.ddim_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0) -> dict` returns `{"sample": Tensor, "pred_xstart": Tensor}` for one DDIM transition at timestep `t`; `sample` is the transitioned latent and `pred_xstart` is the clean-latent estimate, both shaped like `x`.
- `GaussianDiffusion.dfot_sample_loop(model, shape, scheduling_matrix, context=None, n_context_frames=0, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0) -> Tensor` consumes the supplied timestep matrix and context and returns the requested latent-video `shape`.
- `GaussianDiffusion.dfot_ddim_sample(model, x, curr_t, next_t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0) -> dict` returns `{"sample": Tensor, "pred_xstart": Tensor}` for a transition from `curr_t` to `next_t`, with transitioned latent and clean-latent estimate both shaped like `x`.
- `FlowMatching.dfot_sample_loop(model, shape, scheduling_matrix, context=None, n_context_frames=0, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0) -> Tensor` has the same input and returned-shape contract as the Gaussian loop, for the Flow Matching backend.
- `FlowMatching.dfot_ddim_sample(model, x, curr_t, next_t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0) -> dict` returns the Flow Matching transition's `sample` and `pred_xstart`, both shaped like `x`.

The sampler returns to the trusted caller; it does not write files. The caller decodes the tensor, clamps decoded RGB values to `[-1, 1]`, and rescales them to `[0, 1]`. The retained writer creates one NumPy archive per clip, with exactly these fields and axis orders:

```text
<run-directory>/eval_videos/step_0000000/<clip-tag>/raw_data.npz
  pred: float32 array[3, 4, 256, 256] in [0, 1]  # C, T, H, W
  gt:   float32 array[3, 4, 256, 256] in [0, 1]  # C, T, H, W
```

`<clip-tag>` has the format `traj_<trajectory:04d>_start_<start:04d>` and is assigned by the caller. For the real input pixel above, the corresponding saved target is `gt[:, 0, 0, 0] = [0.098039, 0.200000, 0.298039]`. The same rescaling applies to predicted pixels: a decoded RGB value of `[-1.0, 0.0, 1.0]` at a future-frame pixel is written as `[0.0, 0.5, 1.0]` in `pred` at the corresponding location. These examples specify the data conversion; there is no single prescribed predicted pixel value for a clip.

The evaluator reads `pred[:, 1:]` and `gt[:, 1:]`, producing three `(3, 256, 256)` predicted/target frame pairs. A structurally valid run must produce exactly one such archive for each requested clip. The target archive content comes from the loaded RGB video, not a decoder reconstruction. The writer also creates `pred_video.mp4` and `gt_video.mp4` visualizations, but scoring reads the float arrays. Temporary run files are removed after measurement.

The evaluator invokes the retained program with every override shown below. `python` denotes the verifier's Python executable, and placeholders denote absolute paths selected by the verifier:

```text
python src/main.py experiment=evaluate_only dataset=dino_wm/point_maze model=nanowm_b2 experiment.resume_from_checkpoint=<pinned-checkpoint> dataset.loader.validation_fixed_subset_path=<suite-manifest> experiment.evaluation.batch_size=1 experiment.evaluation.save_videos=true experiment.infra.num_workers=0 experiment.infra.compile=false experiment.infra.mixed_precision=false wandb.enabled=false logger.name=none hydra.run.dir=<temporary-run-directory>
```

The caller's environment supplies `DATASET_DIR`, `VAE_MODEL_PATH`, and `RESULTS_DIR` for the fixed data, VAE, and temporary output locations; it sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `PYTHONHASHSEED=3407`. The checkpoint is the pinned released Point Maze model, converted losslessly from safetensors to the tensor mapping accepted by the retained loader. The sampler is called through the isolation boundary described above, so the native caller's dataset access and environment are not additional sampler inputs.

## Validity
Patch validation is first: the submitted patch must be nonempty text without NUL or binary patch content, apply cleanly to the pristine tree, touch only the literal editable paths, contain no more than 20 changed paths, and be at most 262,144 bytes. A rename counts both its source and destination; a copy counts its destination. The three fixed files and every file under `src/diffusion/frontier_solution/` must be regular, nonsymlinked `.py` or `.pyi` source that parses and compiles as Python. Each directory name and filename stem in a source path must be a valid Python identifier. The runtime imports `.py` modules; `.pyi` files are not executable helper modules.

Static policy is applied to the resulting editable source before execution. Absolute imports are limited to these roots: `diffusion`, `collections`, `dataclasses`, `enum`, `functools`, `itertools`, `math`, `numpy`, `operator`, `torch`, `tqdm`, and `typing`. Case-insensitive source text may not contain `judge_data`, `frontier_submission_role`, `frontier_model_root`, `validation_fixed_subset`, `torch.load`, `numpy.load`, `np.load`, `getenv`, or `environ`. The names `__builtins__`, `__import__`, `builtins`, `compile`, `eval`, `exec`, `getattr`, `globals`, `input`, `locals`, `open`, `setattr`, and `vars` are forbidden. Runtime/frame/file attributes `_getframe`, `ag_frame`, `cr_frame`, `f_back`, `f_code`, `f_globals`, `f_locals`, `fromfile`, `genfromtxt`, `gi_frame`, `load`, `loadtxt`, `memmap`, `save`, `tb_frame`, and `with_traceback`, as well as every dunder attribute access, are forbidden.

The sampler runs in a fresh isolated process with the pinned public model, compact context and action tensors, and numerical libraries initialized before your code runs. New filesystem opens, network access, executable processes and access to other processes are unavailable. Your diffusion modules and Python helpers are imported from source in memory, with at most 2 MiB of `.py` source across the diffusion package. A later import cannot read an additional file from disk. The sampler returns only a finite float32 tensor of the requested shape; the trusted process retains the targets, decoded-output writer and quality metrics. The source checks are additional policy checks, not the isolation boundary.

Patch rejection, source/build failure, policy violations, and candidate crashes, timeouts, or out-of-memory failures score zero before quality is considered. After patch and source checks, the verifier establishes a successful trusted comparison run before executing the candidate. Wrong tensor type, shape, dtype or device, non-finite tensor values, incomplete output, wrong result count, or malformed or non-finite metrics also score zero. The required result count is the suite's clip count (four in development, six in the graded suite), each with exactly the numeric metric keys `psnr`, `ssim`, and `lpips`. The trusted caller computes these values; the sampler cannot submit them directly.

Failures of the trusted comparison run, native caller or writer, required assets, or isolation infrastructure fail verification without a measured candidate quality score. Diagnostics remain private. Scoring also requires an execution audit showing that the retained `src/main.py` reached its experiment dispatch, checkpoint loader, real Point Maze frames, codec, sampling integration, and prediction writer. Each native program invocation has a 1,650-second timeout. Its isolated calls share the remaining time under a deadline set ten seconds before that limit, including caller setup and earlier calls. Resource limits are listed under Resources.

## Scoring
The goal is to maximize the score on a 0–100 scale.

PSNR (peak signal-to-noise ratio) measures pixel fidelity in decibels. SSIM (structural similarity) compares local image structure, and LPIPS (learned perceptual image patch similarity) measures distance between learned image features.

For clip `i`, let `P_i` be the arithmetic mean of per-frame PSNR in dB, `S_i` the arithmetic mean of per-frame SSIM, and `L_i` the arithmetic mean of per-frame LPIPS over its three future frames. PSNR is computed on `[0,1]` RGB values with value range `1.0` and epsilon `10^-8`; SSIM uses an 11×11 window, sigma `1.5`, and three channels; LPIPS uses the VGG network after mapping RGB values from `[0,1]` to `[-1,1]`. Define

```text
C(x) = min(1, max(0, x))

Q_i = 100.0 * (
        C((P_i - 20.0) / 30.0)
      + C((S_i - 0.80) / 0.20)
      + C((0.25 - L_i) / 0.25)
      ) / 3.0

candidate_score = (Q_1 + Q_2) / 2      # the suite's two original clips
score = min(100.0, max(0.0, candidate_score))
```

The constants `20.0`, `30.0`, `0.80`, `0.20`, and `0.25` are challenge constants (not from materials). Higher PSNR and SSIM increase the score; lower LPIPS increases it. Each metric is clipped separately for each clip before the clip scores are averaged. The score is not computed by applying this formula to the suite-wide metric summaries. Passing the separate comparison criterion is not a multiplier or gate on this numerical score.

## Calibration
The shipped sampler reaches `NotImplementedError`, produces no valid prediction, and scores zero. The task has no measured A10G quality result for a functioning naive implementation and no empirical run-to-run noise floor or significance threshold. Fixed clips and seed 3407 define the evaluation protocol, but there is no measured repeatability bound from which to declare a small score change significant. The separate per-clip comparison has no tolerance: equality on a metric does not pass its strict comparison.

For an arithmetic calibration of the score scale, if both original clips had `PSNR=30.0`, `SSIM=0.90`, and `LPIPS=0.125`, each would contribute normalized terms `1/3`, `1/2`, and `1/2`, giving a score of approximately `44.4444`. This is a hypothetical metric record, not a measured sampler result. While the clipping bounds are inactive, a 1 dB PSNR increase on one of the two original clips changes the suite score by `100 / (3 * 30 * 2)`, approximately `0.5556` points. These numbers explain the score's units; they are not quality targets or experimental evidence. There is no designated stopping threshold below the maximum score.

## Feedback and grading
Each train call returns a score and the development metrics once its verification completes; saving or submitting the final answer returns no score to you. The submission budget is in `/frontier/task/README.md`. The graded suite is not the suite used for development feedback: they contain disjoint clips, and neither suite's clip membership nor suite-construction seeds are disclosed. They are stratified so they carry the same distribution and share no item.

A separate, stricter quality criterion is evaluated per held-out Point Maze clip and must hold on every graded clip, including the clips added by beat spec v3 (four development, six graded). It compares the candidate with a hidden comparison implementation on the same three future frames. The criterion is satisfied only when, on every clip, candidate PSNR and SSIM are both strictly higher and candidate LPIPS is strictly lower. The numerical score remains the optimization objective; this separate criterion does not replace it, and a submission that does not satisfy the criterion is still scored normally.

Development feedback exposes the candidate's arithmetic-mean PSNR, SSIM, and LPIPS across the four development clips. For each of those metrics it also reports how many development clips passed and failed the hidden per-clip comparison, the total, and the corresponding pass/fail percentages. It does not expose clip identities, per-clip rows, hidden values, ratios, differences, or graded-suite metrics.

## Resources
The configured limits are:

| Resource | Limit |
| --- | --- |
| Container CPUs | 3 vCPUs |
| Container memory | 12,288 MiB |
| Workspace storage | 10,240 MiB |
| Build timeout | 600 seconds |
| Runtime timeout | 10,200 seconds |
| Agent accelerator | One NVIDIA A10G instance; at most 1,800 instance-seconds and 1,800 GPU-seconds |
| Verifier accelerator | One NVIDIA A10G instance; at most 10,200 instance-seconds per evaluation and 20,400 GPU-seconds in total |
| Each native evaluation invocation | 4,950 seconds, including its isolated sampler calls (all clips of the suite) |
| Each isolated worker | 1,650 CPU-seconds, additionally subject to the caller's remaining wall-clock deadline |

Verification uses Python 3.11, PyTorch 2.4.1, torchvision 0.19.1, NumPy 1.26.4, diffusers 0.30.3, PIQA 1.3.2, and LPIPS 0.1.4 in the configured CUDA 12.4 runtime. The Point Maze checkpoint, SD-VAE, and VGG metric weights are preinstalled, and verification is offline. Resource limits and timeout values are challenge constants (not from materials); no measured GPU peak-memory requirement or runtime calibration is available.

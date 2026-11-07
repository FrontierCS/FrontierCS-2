# Sparse Video Attention Policy

Make text-to-video generation faster without losing fidelity. You implement the sparse
self-attention policy used by two video diffusion models. Each clip is scored on how closely its
video matches the dense-attention video, and on how much faster it is generated.

## Task

Sparse VideoGen (built on Hugging Face Diffusers) runs two text-to-video paths: Wan2.1-T2V-14B and
HunyuanVideo. Both denoise spatiotemporal tokens with transformer self-attention. In this tree the
EAR attention processor classes of both models and three shared tensor helpers are stubbed
(`NotImplementedError`). Reconstruct them, reuse the retained sparse-attention classes, or write
another policy. Both models must run: a HunyuanVideo workload whose processor cannot be built fails
like a failed Wan workload.

## What you may change

- `svg/models/wan/attention.py`: `WanAttn_EARAttn_Processor`
- `svg/models/hyvideo/attention.py`: `Hunyuan_EARAttn_Processor`
- `svg/kmeans_utils.py`: `identify_dynamic_map_estimated`, `scatter_mean_fused`,
  `dynamic_block_sparse_prune_fwd_flashinfer`
- new or changed modules under `svg/solver_extensions/`, imported from the files above.

These five symbols must stay **defined directly** in their files; importing them from another file
fails the check. Everything else is fixed: pipelines, checkpoints, text conditioning, schedulers
and denoising schedules, the transformers and VAEs outside these files, the retained cross-attention
processor, MP4 export and decoding, the timing boundary, and the metrics.

Added code must not contain the words `inspect`, `subprocess`, `runpy` or `open` (matched as
whole words), nor anywhere, even inside a longer name, `__closure__`, `__dict__`, `__import__`,
`_getframe`, `/proc/`, `perf_counter` (so `perf_counter_ns` too), `WanPipeline`,
`HunyuanVideoPipeline` or `sys.modules`. A violation is rejected as `content_policy`.

## Interface

Before the scored call the judge rebuilds every processor as
`type(p)(layer_idx=p.layer_idx)`, so both classes must accept `layer_idx` as a keyword and keep
it as an attribute.

**Wan.** The processor is built as `WanAttn_EARAttn_Processor(layer_idx: int)` and called as
`processor(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None,
timestep=None) -> Tensor`. It must accept every argument. On the scored path `hidden_states` is
CUDA bfloat16 `(1, 75600, 5120)`, `encoder_hidden_states` and `attention_mask` are `None`,
`rotary_emb` is given, and `timestep` is a one-element CUDA tensor. The return has the shape,
device and dtype of `hidden_states`. Q/K/V are `(B, H, L, D) = (1, 40, 75600, 128)`. Wan has 40
layers and 40 heads, and 21 latent frames of 45 × 80 tokens give L = 75,600.

**HunyuanVideo.** The processor is built as `Hunyuan_EARAttn_Processor(layer_idx: int)` for all 60
layers: dual-stream layers 0–19 and single-stream layers 20–59. Before construction the
integration sets the class attributes `prompt_length`, `context_length=256`, `num_frame=21`,
`frame_size=3600`, `first_layers_fp`, `first_times_fp`, the k-means settings and `logging_file`.
The call is `processor(attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
image_rotary_emb=None, timestep=None) -> (video, text)`, where the inputs are:
- `hidden_states` is the video stream, CUDA bfloat16 `(1, 75600, 3072)`.
- `encoder_hidden_states` is the text stream, `(1, 256, 3072)`. Dual-stream layers project it with
  the module's added projections; single-stream layers use the video projections
  (`attn.add_q_proj is None`).
- `attention_mask` is boolean `(1, 1, 1, 75856)`: true for video and prompt tokens, false for text
  padding.
- `image_rotary_emb` is the `(cos, sin)` pair for the video tokens.
- `timestep` is a one-element CUDA tensor.

Both outputs keep their input's shape, device and dtype. Attention is joint over video then the
256-slot text context, so Q/K/V are `(1, 24, 75856, 128)`.

**Helpers** (`svg/kmeans_utils.py`). Every size row sums to L. Each function must return:

```python
identify_dynamic_map_estimated(q_perm, k_perm, v_perm, qc_sz, kc_sz,
        qcentroids, kcentroids, vcentroids, top_p=0.9, gamma=1.0, min_kc_ratio=0.0)
    # q/k/v_perm (B,H,L,D), tokens contiguous by cluster; qc_sz (B,H,Cq), kc_sz (B,H,Ck) int CUDA;
    # centroids (B,H,Cq,D), (B,H,Ck,D), (B,H,Ck,D)  ->  CUDA bool (B,H,Cq,Ck)
scatter_mean_fused(src, size)
    # src contiguous (B,H,L,D); size (B,H,K) consecutive groups  ->  (B,H,K,D), mean per group,
    # zeros for an empty group, same device/dtype as src
dynamic_block_sparse_prune_fwd_flashinfer(q, k, v, k_centroids, v_centroids,
        block_mask_map, block_row_sz, block_col_sz, is_cpu=True)
    # q/k/v (B,H,L,D); centroids (B,H,Ck,D); map bool (B,H,Cq,Ck); row/col sizes (B,H,Cq)/(B,H,Ck);
    # is_cpu=True: map and sizes on CPU, else may stay on CUDA  ->  same shape/device/dtype as q
```

```python
>>> scatter_mean_fused(torch.tensor([[[[1., 0.], [3., 2.], [8., 4.]]]], device="cuda"),
...                    torch.tensor([[[2, 1]]], device="cuda"))
tensor([[[[2., 1.], [8., 4.]]]], device="cuda:0")
```

**Entrypoints.** The judge runs `bash scripts/wan/wan_t2v_720p_ear.sh` and
`bash scripts/hyvideo/hyvideo_t2v_720p_ear.sh`. It replaces the prompt, the seed, the offline
model path and the output and logging paths, and sets `--pattern EAR` and `--num_frames 81`
(HunyuanVideo's parser default is 129). The scripts fix the remaining settings:
- Wan: `--pattern EAR`, 300 q / 1000 k centroids, `top_p_kmeans 0.9`, `min_kc_ratio 0.10`,
  k-means iterations 50 then 2 per step, `first_times_fp 0.2`, `first_layers_fp 0.03`.
- HunyuanVideo: the same but 400 q centroids, `--zero_step_kmeans_init`, `first_times_fp 0.1`.

Each run writes the MP4 given by `--output_file`. The `--logging_file` output is not read or
scored.

## Correctness

Any of these on any workload makes the whole evaluation score 0:
- the offline checkpoint is missing or incomplete;
- Python syntax or import fails, the CUDA extension fails, or generation crashes;
- a timeout: extension build 600 s, generation process 3,600 s, workload worker 3,900 s;
- latency is non-positive or non-finite;
- the MP4 is missing or malformed, or does not decode to exactly `(81, 720, 1280, 3)` uint8 RGB;
- a fidelity value is invalid: PSNR `NaN` or −∞, or non-finite SSIM or LPIPS.

PSNR +∞ is valid for an exact match.

## Workload

One workload is one fixed (model, prompt, seed) clip at 720 × 1280, 81 frames and 50 denoising
steps. It is compared with a fixed dense-attention video of the same model, prompt and seed. The
dense video is not regenerated from your code.

| suite | Wan clips | HunyuanVideo clips |
|---|---|---|
| train (4) | prompts 1, 2 (one seed each) | prompts 3, 4 (one seed each) |
| final (12) | prompts 1–4, two seeds each | prompts 1–4, one seed each |

- The two suites share no (model, prompt, seed) item, and every seed is distinct. Seeds (train and
  final) and final identifiers are withheld.
- Four workloads run at a time, each on its own H100. Train is one wave; final is three waves: Wan,
  Wan, then HunyuanVideo.
- Prompts are `examples/{1,2,3,4}/prompt.txt`. Each file is one ASCII record of 680, 678, 1,064 or
  581 bytes, with no trailing newline. `examples/7` is read by the HunyuanVideo script but is not a
  workload.
- Wan uses `guidance_scale=5.0` with classifier-free guidance and the parser's default negative
  prompt. HunyuanVideo uses embedded guidance 6.0 (one transformer pass per step), a flow-matching
  Euler scheduler with shift 7.0, and 81 frames (upstream uses 129).
- Videos are exported at 16 fps for Wan and 24 fps for HunyuanVideo.

Each workload calls the pipeline twice with its seed. First comes an unmeasured 2-step warm-up,
whose frames are discarded. The judge then builds new processor instances and empties the
HunyuanVideo class-level k-means caches (`centroids_init`, `q_centroids`, `k_centroids`). The second
call runs all 50 steps, with the GPU synchronized just before and after; that interval is
`e2e_latency_seconds`. Model loading, processor replacement, the extension build, MP4 export and
decoding, and fidelity evaluation are not timed.

## Scoring

Per workload, with `T` the candidate latency and `Td` the fixed, pre-measured dense latency, and with `P`, `S` and `L`
the means over the 81 frames of PSNR, SSIM and VGG-LPIPS against the dense video:

```text
clip(x) = max(0, min(1, x))
quality    = (clip((P - 15.0)/20.0) + clip((S - 0.50)/0.50) + clip((0.35 - L)/0.35)) / 3
efficiency = clip(Td / max(1e-12, T) - 1.0)
score      = 100 * quality * (0.25 + 0.75 * efficiency)
```

The metric definitions use frames converted to [0, 1]:
- PSNR is `10·log10(1/MSE)`, with MSE over all pixels and channels.
- SSIM uses an 11 × 11 average-pool window (stride 1, padding 5) and constants 0.01² and 0.03².
- LPIPS is the VGG variant on the [0, 1] frames.

The suite score is the mean over its workloads: 4 in train, 12 in final. Anchors: a video identical
to the dense one at `T = Td` scores 25; at `Td/T ≥ 2` it scores 100. The untouched tree cannot run
because the stubs raise `NotImplementedError`.

**Beats the reference**: on every one of the 12 final workloads, both of these hold against the
withheld reference implementation:
- latency is at least 4% lower;
- PSNR is at most 0.25 dB lower, SSIM at most 0.01 lower, and LPIPS at most 0.005 higher.

Equality at a bound passes. This criterion does not change the score.

## Running locally

See `/frontier/task/README.md` (`frontier-build.sh` and `frontier-bench.sh [hunyuan]` on a GPU
runner).

## Environment

The fixed runtime is:
- Python 3.11, PyTorch 2.6.0, CUDA 12.4, and H100 (SM90a) GPUs.
- Diffusers 0.34.0, FlashInfer 0.2.10, cuVS 26.2.0, Triton 3.2.0, imageio 2.37.0 and LPIPS 0.1.4.
- Checkpoints, offline: `Wan-AI/Wan2.1-T2V-14B-Diffusers@38ec498c` and
  `tencent/HunyuanVideo@2a15b557` (Tencent Hunyuan Community License).

## Notes

The workloads are fixed generations representative of the two paths, not strings to recognize:
solutions must not special-case prompts, seeds or the train/final phase.

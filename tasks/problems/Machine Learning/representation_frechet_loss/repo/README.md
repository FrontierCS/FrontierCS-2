# Differentiable Population Statistics

Implement `frechet_distance.queue.FeatureQueue`, the differentiable population mean/covariance
statistics with which an image generator is post-trained against representation Fréchet
distances, so that the post-trained generator reaches lower held-out normalized Fréchet distances
in six representation spaces. A valid submission keeps the `FeatureQueue` contract and passes the
trusted checks below. A beats-reference criterion is evaluated alongside the score (see Scoring).

## Task

This repository ("Representation Fréchet Loss for Visual Generation") post-trains a
class-conditional pMF image generator with Fréchet distances measured by frozen representation
models. Training reads generated feature batches together with persistent population state built
from generated images; `FeatureQueue` turns them into population mean/covariance estimates or
feature snapshots that stay differentiable with respect to the current batch and survive ordinary
PyTorch checkpoints. The method bodies of `FeatureQueue` have been removed.

## What you may change

- `frechet_distance/queue.py`
- new or changed modules below `frechet_distance/frontier_solution/`, used through `FeatureQueue`.

At most 20 files may change; everything else is fixed: generator sampling, representation
extraction, the Fréchet algebra, optimizer behaviour, checkpoint orchestration, `main_fd.py`,
`utils/builders.py`, `models/denoiser_pmf.py` and `frechet_distance/judges.py`.

The judge runs `FeatureQueue` in a separate isolated process in which only
`frechet_distance/queue.py` and `frechet_distance/frontier_solution/` are importable, with no
network, subprocesses or file access and one CPU thread; tensors are passed by value. Imports of
other repository modules (`frechet_distance.metrics`, `utils.*`), `torch.distributed` or file
I/O work under `frontier-bench.sh` but fail in the judge.

## Interface

`FeatureQueue(size: int, feat_dim: int, online_accum: bool = False, ema_beta: float = 0.0)` keeps the shipped constructor's buffers: in EMA mode `mu_ema` `(feat_dim,)` float64, `m2_ema` `(feat_dim, feat_dim)` float64 and `_ema_count` `(1,)` int64; otherwise `feats` `(size, feat_dim)` float32 and `ptr` `(1,)` int64, plus `feat_sum_old` and `feat_outer_old` float64 in online mode. `build_feats_stats(new_feats)` returns `(mu, sigma)` with shapes `(feat_dim,)` and `(feat_dim, feat_dim)`. `build_feats_snapshot(new_feats)` returns a `(size, feat_dim)` tensor. `enqueue(new_feats)` updates persistent state without retaining an autograd graph. The initialization methods used by `frechet_distance/judges.py` must remain callable.

For example:

```python
q = FeatureQueue(size=4, feat_dim=2)
q.feats.copy_(torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]))
x = torch.tensor([[2., 3.]], requires_grad=True)
s = q.build_feats_snapshot(x)
tuple(s.shape)                 # (4, 2)
s[0].tolist()                  # [2.0, 3.0]
```

`FeatureQueue` must work in snapshot, online-accumulator (`online_accum=True`) and EMA
(`ema_beta > 0`) modes; every retained caller must keep working.

## Correctness

Score 0 unless all of these hold on every workload:
- the implementation runs through the unchanged `main_fd.py` on the assigned GPU, and each
  `FeatureQueue` call (the first one including importing your module) returns within 15 s;
- returned means, covariances and snapshots have the declared shapes, stay differentiable with
  respect to the current batch where training calls them, and are finite, as are features,
  losses and gradients;
- the buffers keep their names, shapes and dtypes, stay finite, `ptr` stays in `[0, size)` and
  `_ema_count >= 0`; state updates without retaining an autograd graph and round-trips through a
  `state_dict` save/load;
- the generator update is genuine (finite, nonzero gradients, changed parameters);
- the generated warm start is authentic and both sides' held-out sample counts are equal;
- your train or eval phase does not fail or exceed the suite time limit.

## Workload

Every workload starts from the same released base pMF-B generator at 256 pixels and uses the unchanged FD-SIM training path. It builds a 50,000-image generated warm-start population, performs a bounded post-training schedule with batch size 8, and evaluates a 4,096-image held-out generated set per side in Inception, ConvNeXt, DINOv2, MAE, SigLIP, and CLIP spaces. Workloads vary fixed random streams, update counts, and queue estimator settings; they include 128- and 512-update schedules. ImageNet images, statistics regeneration, precision/recall, visualization, compilation, and online evaluation are outside the workload.

| suite | workloads |
|---|---|
| train | 2 |
| final | 3 |

The suites are disjoint; final workloads are withheld. The train suite runs within 7,200 s.

## Scoring

For representation `r`, `FDr_r = held_out_FD_r / validation_FD_r`, with the published
validation-set normalizers 1.68 (Inception), 56.87 (ConvNeXt), 14.19 (DINOv2), 0.04 (MAE), 0.60
(SigLIP) and 5.60 (CLIP). Per workload `FDr6` is the mean of the six `FDr_r`:

```text
score = clip(100 / (1 + mean over workloads of FDr6), 0, 100)
```

Each workload has equal weight regardless of its update count or estimator setting. Anchors:
`FDr6 = 0` scores 100, `FDr6 = 1` scores 50.

**Beats the reference** when every validity check passes and, on every final workload, all six
`FDr_r` are strictly lower than the reference's. It does not change the score.

## Running locally

On a GPU runner, `bash frontier-build.sh` builds the tree. `frontier-bench.sh` prepares the
offline assets (`frontier_prepare_assets.py` converts the hash-checked public bundle to `.npz`
files and links the staged weights below `/workspace/fd-assets`; paths and revisions are in
`frontier_public_assets.json`), supplies Inception statistics defaults (explicit arguments override
them) and starts the unchanged `main_fd.py`. A full generated warm start and one update:

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime \
  bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime \
  bash /workspace/repo/frontier-bench.sh \
  --model pMF_B --img_size 256 --patch_size 16 --rope_2d --learned_pe \
  --disable_v_head --dtype fp32 \
  --load_from /opt/frontier/models/fd-loss/checkpoints/base/pMF-B_256.pth \
  --batch_size 8 --queue_size 50000 --fd_queue_fill_bsz 8 --fd_eigvalsh \
  --fd_ema_beta 0.999 \
  --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception \
  --fd_repr_pool_types cls cls cls --fd_target_sizes 224 224 256 \
  --fd_repr_stats_paths \
    /workspace/fd-assets/stats/vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz \
    /workspace/fd-assets/stats/vit_large_patch16_224_mae_in256_t224_stats.npz \
    /workspace/fd-assets/stats/guided_diffusion_stats.npz \
  --num_sampling_steps 1 --epochs 1 --steps_per_epoch 1 \
  --warmup_epochs 0 --lr 1e-6 --lr_sched constant --min_lr 0 \
  --disable_vis --vis_freq 0 --eval_freq 0 --disable_wandb \
  --milestone_interval 0 --save_freq 1 --print_freq 1 \
  --output_dir /workspace/public-check
```

## Environment

`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, Python 3.10; one NVIDIA A100 80GB. The released
checkpoint, statistics bundle and representation weights are offline under `/opt/frontier/models`.

## Notes

Workloads are generated, not records to recognise.

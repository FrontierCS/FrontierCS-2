# Differentiable Population Statistics

> **Source revision awaiting runtime qualification.** Submitted queue code now
> has an isolated execution route. The resulting verifier image and complete
> scientific workloads still require validation before this revision can be
> used as a solver benchmark.

## Problem
This repository post-trains a class-conditional image generator using Fréchet distances measured by frozen representation models. It reads generated feature batches together with persistent population state built from generated images. It produces population mean/covariance estimates or feature snapshots that remain differentiable with respect to the current batch, and it must preserve that state through ordinary PyTorch checkpoints. The generator update must have finite nonzero gradients and all reported features and covariances must be finite. Quality is measured after post-training by held-out normalized Fréchet distances in six representation spaces, where lower is better. The implementation bodies of `frechet_distance.queue.FeatureQueue` have been removed and must be supplied.

## Required behavior
The implementation must satisfy the `FeatureQueue` constructor and method
contracts below for snapshot, online-accumulator, and EMA modes. Returned means,
covariances, and snapshots must have the declared shapes, remain differentiable
with respect to the current batch where called by training, and stay finite.
Persistent state must update without retaining an autograd graph and must
round-trip through ordinary PyTorch `state_dict` save/load. Every retained
caller must continue to work. Helper code may live in the fresh solution
namespace, but the integration surface remains `FeatureQueue`. Generator
sampling, representation extraction, Fréchet algebra, optimizer behavior, and
checkpoint orchestration are outside the editable surface.

## Workload
Every workload starts from the same released base pMF-B generator at 256 pixels and uses the unchanged FD-SIM training path. It builds a 50,000-image generated warm-start population, performs a bounded post-training schedule, and evaluates an equal-count held-out generated set in Inception, ConvNeXt, DINOv2, MAE, SigLIP, and CLIP spaces. Workloads vary fixed random streams, update counts, and queue estimator settings; they include 128- and 512-update schedules. ImageNet images, statistics regeneration, precision/recall, visualization, compilation, and online evaluation are outside the workload. These are generated workloads rather than records to recognize. The source defines exactly two development and three final workloads.

## Interface
`FeatureQueue(size: int, feat_dim: int, online_accum: bool = False, ema_beta: float = 0.0)` must retain its registered-buffer/state-dict surface. `build_feats_stats(new_feats)` returns `(mu, sigma)` with shapes `(feat_dim,)` and `(feat_dim, feat_dim)`. `build_feats_snapshot(new_feats)` returns a `(size, feat_dim)` tensor. `enqueue(new_feats)` updates persistent state without retaining an autograd graph. The initialization methods used by `frechet_distance/judges.py` must remain callable.

For example:

```python
q = FeatureQueue(size=4, feat_dim=2)
q.feats.copy_(torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]))
x = torch.tensor([[2., 3.]], requires_grad=True)
s = q.build_feats_snapshot(x)
tuple(s.shape)                 # (4, 2)
s[0].tolist()                  # [2.0, 3.0]
```

## Public GPU checks

After implementing the queue, upload your working repository to the Agent GPU
runner as described in `/frontier/task/README.md`. The image already contains
the released checkpoint, statistics bundle and representation model weights
under `/opt/frontier/models`; no download or dataset access is needed.
`frontier_prepare_assets.py` converts the hash-checked public bundle to `.npz`
files and links the staged weights into an offline Hugging Face cache below
`/workspace/fd-assets`. It can be run by itself to inspect setup without training.
The paths and revisions are listed in `frontier_public_assets.json`.

Use the image's Python and restored environment, including over non-login SSH:

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

This example exercises a full generated warm start and one update; allocate
enough runner time for that work. You may choose other public debug parameters.
It is not a development score or the private schedule. `frontier-bench.sh`
prepares offline assets and supplies Inception statistics defaults; explicit
arguments override those defaults. It starts the unchanged `main_fd.py` through
the runtime interpreter's distributed launcher. The Verifier independently
supplies its declared workload and statistics paths.

## Validity contract
The evaluator rejects patches outside the editable surface and requires
the submitted implementation to execute through `main_fd.py` on the assigned
NVIDIA GPU. Before quality is scored, trusted checks must establish locked
source/model identity, finite features/covariances/losses/gradients, a genuine
generator update, queue-state round-trip, an authentic generated warm start,
and equal candidate/reference held-out sample counts. The trusted runner owns
training and evaluation, and calls the submitted queue in a separate isolated
process. This revision's complete scientific execution remains unqualified.

## Scoring
For each workload, let `FDr6` be the arithmetic mean of the candidate's six held-out FDr values. The score to maximise is

```text
score = 100 / (1 + mean_workloads(FDr6))
```

and is clipped to `[0, 100]`. The six published normalization constants are 1.68 (Inception), 56.87 (ConvNeXt), 14.19 (DINOv2), 0.04 (MAE), 0.60 (SigLIP), and 5.60 (CLIP).

For a representation `r`, `FDr_r = held_out_FD_r / validation_FD_r`; all terms are nonnegative and lower candidate values produce a higher score. Each workload contributes exactly one `FDr6` regardless of its update count or estimator setting, and workloads receive equal weight in the outer arithmetic mean. Invalid submissions receive zero rather than entering this formula. The returned unbounded score equals the bounded score. Development feedback reports the candidate-only arithmetic mean for each of the six `FDr_r` quantities, so those values can be compared between your own submissions without exposing paired measurements from the withheld implementation.

## Calibration status
No admissible non-reference baseline has reached quality scoring. Historical
published-runtime evidence contains one development measurement and one final
measurement only; it is not a repeatability study. Paired results and final
measurements remain private author evidence. No zero-noise claim, tolerance,
meaningful-improvement band, or repeated stability claim is supported. The
strict comparisons and scientific workloads remain unchanged. Any further
measurement must address a specific uncertainty in those comparisons; automatic
repeated reference trials and a beating solver are not prerequisites. This
candidate awaits validation of the integrated verifier and full workloads.
The available local CPU checks are boundary and contract tests, not scientific
quality calibration.

## Feedback and grading
The configured limit is 3 development submissions. Development
feedback includes only the candidate's
mean absolute FDr for each representation and aggregate pass/fail percentages;
it must not identify workload rows or reference values. Development and final
suites contain two and three disjoint workloads respectively. These feedback
paths still require validation on this revision's verifier image.

In addition to the continuous score, a separate stricter criterion compares your held-out generated-image FDr independently in Inception, ConvNeXt, DINOv2, MAE, SigLIP, and CLIP space against the reference on the same private workload. Lower is better for every quantity, and the stricter criterion is met only if your value is strictly lower in all six representations on every graded workload and all validity checks pass. Values from the withheld implementation, paired differences, private workload identities, per-workload rows, and all final measurements are not disclosed. The continuous score remains the optimization objective whether or not this stricter criterion is met.

## Resources
The control pod has 3 vCPUs, 12,288 MiB RAM, and 10,240 MiB workspace storage, with a 900-second build limit. Agent and verifier work use one Driver-provisioned A100 80GB GPU. The answering window is 18,000 seconds. Each train verification has a 7,200-second limit, with three calls and 21,600 train seconds in total. Independent final evaluation has its own 10,800-second limit. Agent debugging has a separate 7,200-second GPU allowance; the verifier GPU pool is 32,400 seconds.

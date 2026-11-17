# Learned Transformer Layer Mapping

Implement Dynamic Tanh (DyT), the element-wise replacement for every `LayerNorm` in a ViT, so
that the converted models compute what the authors' DyT computes, and make training and inference
faster than the authors' implementation (the score counts training). A valid submission converts every site and passes every
numerical check below. **First priority: beat the reference on every final workload and metric
(see Scoring); the latency score is secondary.**

## Task

This repository is the authors' image-classification trainer for ViT models with DyT
("Transformers without Normalization", CVPR 2025). With `--dynamic_tanh true` the unchanged
`main.py` builds a timm model, passes it to `convert_ln_to_dyt`, which replaces every
`LayerNorm` with a learned DyT site, then trains and evaluates it. The implementation in
`dynamic_tanh.py` has been removed.

## What you may change

- `dynamic_tanh.py`
- new or changed modules below `solver_extensions/`, imported from `dynamic_tanh.py`.

At most 32 files may change; everything else is fixed. No file you own (`dynamic_tanh.py` and
everything below `solver_extensions/`, comments included) may contain, case-insensitively, `argv`,
`__main__`, `inspect`, `_getframe`, `currentframe`, `f_back`, `f_globals`, `f_locals`, `f_code`,
`traceback`, `getpid`, `getppid`, `/proc`, `environ`, `getenv`, `sys.modules`, `cmdline` or
`program_invocation` (`inspect` and `traceback` as whole words, the others anywhere; use
`tempfile` for private scratch names); a violation is rejected as
`content_policy`.

## Interface

- `DynamicTanh(normalized_shape, channels_last, alpha_init_value=0.5)` is one site with its own
  learned parameters, named and shaped as the authors' checkpoints: `alpha` of shape `(1,)`,
  initialised to `alpha_init_value`, and `weight` and `bias` of shape `normalized_shape`,
  initialised to ones and zeros. The judge loads the authors' state with
  `load_state_dict(strict=True)`, so other names or shapes (for example a per-channel `alpha`) fail.
- `forward(x)` accepts the tensor layouts of the replaced site (`channels_last` selects the
  channel dimension) and preserves shape and dtype.
- `convert_ln_to_dyt(module)` returns the recursively converted module tree with no
  `torch.nn.LayerNorm` left; sites share no state.

```python
>>> converted = convert_ln_to_dyt(nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)))
>>> sum(isinstance(m, nn.LayerNorm) for m in converted.modules())
0
>>> tuple(converted[1].weight.shape), tuple(converted[1].bias.shape)
((8,), (8,))
```

How you compute it is open: formulation and precision, fused or custom kernels, custom autograd,
compiling the model or the sites, CUDA graphs, specialising by layout or model configuration.

## Correctness

Score 0 unless all hold:
- It runs; each `main.py` training run ends within 180 s (compilation included); no
  CUDA graph capture you began stays open or failed; the current stream stays.
- Every `LayerNorm` is converted; the state loads as under Interface.
- Fresh fp32 model vs the authors': outputs within relative/absolute 3e-4/3e-5, input gradients
  2e-3/2e-5, DyT sites' parameter gradients and one SGD update 3e-3/3e-5.
- Training gives a finite loss and changes the DyT parameters.
- Each warm-up and timed block matches the authors' implementation holding your trained state on
  the same inputs: outputs and loss within 1e-2 relative plus 1e-4 per element (L2); per-step
  gradients, jointly and per group (see Notes); parameters after the block equal the optimizer's
  steps on your gradients.
- Each call computes its own results; one implementation serves every measurement, with no
  behaviour depending on the process, entry point, caller or environment.

## Workload

One workload trains a five-class classifier from scratch for two epochs on 20 flower photographs
and measures top-1 accuracy on 10 others, with `vit_base_patch16_224` or `vit_base_patch32_224`
and batch size 4, 5 or 8, through:

```bash
./frontier-bench.sh --model MODEL --drop_path 0 --batch_size BATCH --epochs 2 --lr 0.004 --min_lr 0.000001 --warmup_epochs 0 --weight_decay 0.05 --data_set image_folder --data_path TRAIN --eval_data_path VAL --nb_classes 5 --dynamic_tanh true --device cuda --num_workers 2 --pin_mem true --dist_eval false --use_amp true --mixup 0 --cutmix 0 --smoothing 0 --aa '' --reprob 0 --model_ema false --save_ckpt true --save_ckpt_num 1 --seed SEED --output_dir OUTPUT
```

| suite | workloads (model, batch) |
|---|---|
| train | 3: patch16/4, patch32/5, patch16/8 |
| final | 3: the same configurations |

The suites share no photograph; final seeds are withheld. The photographs (TensorFlow
`flower_photos.tgz`) are on the GPU runner at `/opt/frontier/public/flower_photos`.

Timed, on both trained models under fp16 autocast, over 16 batches from the validation photographs
with random targets (new per block, shared by both sides):
- training block: 16 steps of forward, cross-entropy, backward, an in-place step of
  `optim_factory.create_optimizer` (AdamW, weight decay 0.05, lr 1e-4) and `zero_grad`, from the
  trained parameters with a fresh optimizer state;
- inference block: 16 forward passes in eval mode without gradients.

Sides alternate: 3 warm-up blocks, then 11 timed. `T_w`, `I_w` (`T*_w`, `I*_w` for the authors',
run as whole-model CUDA graphs) are the medians of the CUDA-synchronized block times.

## Scoring

```text
latency_w = 100 / (1 + (T_w / T*_w)^4)
score     = mean over the workloads of latency_w        (0..100, uncapped ratio)
```

Anchors: the untouched tree raises `NotImplementedError` and scores 0; equal training latency scores 50, a 1.5x faster block about 84, a 2x faster one about 94.
Accuracy and inference latency do not enter the score.

**Beats the reference** when every validity check passes and, on every final workload,
`A_w >= A*_w` (top-1 percentage, no tolerance), `T_w < T*_w` and `I_w < I*_w`.

## Running locally

On a GPU runner: `bash frontier-build.sh` compiles and imports the tree, `python3 public_test.py`
runs a small public contract check, and `./frontier-bench.sh ...` runs the training command above
(see `/frontier/task/README.md`).

## Environment

Python 3.12, PyTorch 2.5.1, torchvision 0.20.1, timm 1.0.15; one NVIDIA A100 80GB.

## Notes

Gradients are checked jointly and separately for `alpha`, `weight`, `bias` and the other
parameters. Any fp16 or fp32 arrangement of the DyT mapping, fused or reordered, passes the gradient check;
missing, zeroed, rescaled, sign-flipped, stale or wrong-input gradients fail. One step per
16-step block may exceed a bound, but never with a wrong shape or non-finite values. The `.grad`
tensors a step leaves must not be overwritten later, so replaying a CUDA graph into the same
gradient buffers (as `torch.compile`'s `reduce-overhead` mode does) fails.

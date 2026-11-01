# Learned Transformer Layer Mapping

## Optimization priority

Your primary objective is **PASS on every required workload and every required
metric**, with all correctness and validity requirements satisfied. The continuous
score is a secondary optimization objective; a higher score cannot compensate for
any failed PASS condition. Do not replace a solution known to satisfy all tested
PASS conditions with one known to fail them merely to increase its score.
Exploratory trials may regress; if no solution passes yet, work on the unmet
conditions and retain a valid submission as a fallback.

Successful verifier execution or a positive score does not mean scientific PASS.
Use the reported per-metric pass/fail progress to assess the tested workloads.
Train feedback covers only its evaluated workloads; passing train does not prove
final PASS. Final independently evaluates the submitted artifact on its specified
suite. The scoring formula, validity rules, and PASS comparisons below are unchanged.

## Problem

This repository is the authors' image-classification trainer for ViT models with Dynamic Tanh (DyT): with `--dynamic_tanh true`, the unchanged `main.py` builds a timm model, passes it to `convert_ln_to_dyt`, which replaces every `LayerNorm` with a learned DyT site, then trains and evaluates it. The implementation in `dynamic_tanh.py` has been removed. Supply it so that the converted models compute what the authors' DyT computes, and make training and inference faster than the authors' implementation with the same results.

## Workspace

You may change only:

```text
dynamic_tanh.py
solver_extensions/**
```

New Python modules below `solver_extensions/` may be imported from `dynamic_tanh.py`. Everything else (entry point, training loop, data loader, optimizer factory, utilities, license and harness scripts) must stay unchanged; the judge applies your patch to a pristine copy and checks those files. `bash frontier-build.sh` compiles and imports the tree, `python3 public_test.py` runs a small public contract check, and `make_submission.sh` packages all in-scope changes, including new files, into the patch you submit and rejects anything outside the two paths. A patch may change at most 32 files and 262,144 bytes.

## Interface

- `DynamicTanh(normalized_shape, channels_last, alpha_init_value=0.5)` is one site with its own learned parameters, named and shaped as the authors' checkpoints: `alpha` of shape `(1,)`, initialised to `alpha_init_value`, and `weight` and `bias` of shape `normalized_shape`, initialised to ones and zeros. The judge loads the authors' state into your converted model with `load_state_dict(strict=True)`, so other names or shapes (for example a per-channel `alpha`) fail.
- `forward(x)` accepts the tensor layouts of the replaced site (`channels_last` selects the channel dimension) and preserves shape and dtype.
- `convert_ln_to_dyt(module)` returns the recursively converted module tree with no `torch.nn.LayerNorm` left.

```python
>>> model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
>>> converted = convert_ln_to_dyt(model)
>>> sum(isinstance(m, nn.LayerNorm) for m in converted.modules())
0
>>> tuple(converted[1].weight.shape), tuple(converted[1].bias.shape)
((8,), (8,))
```

How you compute it is open: numerical formulation and precision, fused or custom kernels, custom autograd, compilation of the whole model or of the sites, CUDA graphs, and specialisation by layout or model configuration, as long as every site is converted and every workload stays correct through the unchanged full model.

## Workload

Each evaluation has three workloads. Each trains a five-class classifier from scratch for two epochs on 20 flower photographs (`daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`) and measures top-1 accuracy on 10 others, with `vit_base_patch16_224` or `vit_base_patch32_224` and batch size 4, 5 or 8. The judge runs, per workload, with its own paths, model, batch size, seed and output directory:

```bash
./frontier-bench.sh --model MODEL --drop_path 0 --batch_size BATCH --epochs 2 --lr 0.004 --min_lr 0.000001 --warmup_epochs 0 --weight_decay 0.05 --data_set image_folder --data_path TRAIN --eval_data_path VAL --nb_classes 5 --dynamic_tanh true --device cuda --num_workers 2 --pin_mem true --dist_eval false --use_amp true --mixup 0 --cutmix 0 --smoothing 0 --aa '' --reprob 0 --model_ema false --save_ckpt true --save_ckpt_num 1 --seed SEED --output_dir OUTPUT
```

The photographs come from TensorFlow's `flower_photos.tgz` (SHA-256 `4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c`, 228,813,984 bytes; CC BY 2.0, `https://creativecommons.org/licenses/by/2.0/`). The GPU runner has the whole archive extracted read-only at `/opt/frontier/public/flower_photos`, in ImageFolder layout (one directory per category, no train/validation split). The graded suite is not the suite you get feedback on: the two share no photograph and have the same distribution, and neither the graded membership nor its seeds are disclosed. Order and repetition of the runs may vary.

## Measurement

For each workload the judge trains your model and the authors' with the command above and records their top-1 accuracies `A_w` and `A*_w`. On the two trained models it then times two blocks, each over 16 batches of the workload's batch size built from its validation photographs with random targets, new for every block and the same for both implementations, under fp16 autocast on the GPU:

- training block: 16 training steps, each a forward pass, cross-entropy, backward pass, an in-place step of the task's optimizer (the retained `optim_factory.create_optimizer`: AdamW with the command's weight decay 0.05, at learning rate 1e-4, about where the training run's schedule ends) and `zero_grad`. Every block starts from the trained parameters with a fresh optimizer state, so the parameters change at every step;
- inference block: 16 forward passes of the trained model in evaluation mode without gradients.

Your blocks and the authors' alternate: three untimed warm-up blocks, then 11 timed ones. `T_w` and `I_w` (and `T*_w`, `I*_w` for the authors' implementation) are the medians of the CUDA-synchronized block times. The authors' model runs its forward and backward passes as whole-model CUDA graphs in these blocks.

## Validity

The score is 0 when any of these fails:

- The patch stays in scope and within the limits above, and no file it owns (`dynamic_tanh.py` and everything below `solver_extensions/`, comments included) contains, case-insensitively, `argv`, `__main__`, `inspect`, `_getframe`, `currentframe`, `f_back`, `f_globals`, `f_locals`, `f_code`, `traceback`, `getpid`, `getppid`, `/proc`, `environ`, `getenv`, `sys.modules`, `cmdline` or `program_invocation` (use `tempfile` for private scratch names).
- The code imports and runs without crashing; each training run through `main.py` finishes within 180 seconds, compilation and warm-up included. When your code returns, no CUDA graph capture it began may be left open or failed, and the current CUDA stream must be unchanged.
- Every `LayerNorm` site is converted, sites share no state, and the state loads as described under Interface.
- On a freshly converted model in fp32, compared with the authors': forward outputs within relative/absolute 3e-4/3e-5, input gradients within 2e-3/2e-5, and the learned parameters' gradients and one SGD update within 3e-3/3e-5.
- Training gives a finite loss and changes the learned DyT parameters.
- Every warm-up and timed block is compared with the authors' implementation holding your trained state, on the same inputs. Each output and loss may differ by 1e-2 relative plus 1e-4 per element (L2 norms). In a training block, the gradients of every step are compared at your parameters of that step, for all parameters together and separately for the sites' `alpha`, `weight` and `bias` and the remaining parameters. They must be as accurate as half-precision arithmetic allows: any arrangement of the mapping in fp16 or fp32, fused or reordered, passes; missing, zeroed, rescaled, sign-flipped or stale gradients, or gradients for other inputs, fail. After the block your parameters must be what the optimizer's steps on your gradients give. The judge keeps the gradient tensors each step leaves in `.grad`, so they must not be overwritten after that step. CUDA graphs are allowed, but each step must produce fresh gradient tensors: replaying a graph into the same gradient buffers, as `torch.compile`'s `reduce-overhead` mode does, fails.

A fast path taken only under the timed conditions must therefore compute correct results for the inputs and parameters of each call; results kept from an earlier call cannot be reused. The same implementation must serve every measurement: the training run through `main.py`, the checks and the timed calls. Behaving differently per measurement, or inspecting which process, entry point, caller or environment the code runs in, is not allowed.

Evaluation uses one NVIDIA A100 80GB and has an 1800-second limit.

## Scoring

As the secondary objective, maximize the returned score from 0 to 100:

```text
latency_w = 100 / (1 + (T_w / T*_w)^4)
score = mean_w(latency_w)
```

Equal latency scores 50, a 1.5x faster training block about 84 and a 2x faster one about 94, with no cap. Accuracy and inference latency do not enter the score.

PASS requires, on every graded workload, `A_w >= A*_w` (top-1 percentage, no tolerance), `T_w < T*_w` and `I_w < I*_w`, with every validity check passing.

## Feedback

Each development call returns, once verified, its score and your own development-suite means of top-1 accuracy, training latency and inference latency, with per-metric workload pass/fail counts. A rejected development submission scores 0 and returns instead a `candidate_diagnostic`: the stage (`patch_policy`, `semantic_validation` for the fresh-model checks, `model_workload`, or `timed_path`) and, for `timed_path`, the block (`training` or `inference`) and the first failing quantity (output, loss, a gradient group, a missing or wrong-shape result, a gradient overwritten after its step, or nonfinite values that do not match), without numbers. A parameter mismatch after a training block is reported as the gradient of all parameters jointly. Reference measurements and per-workload rows are never shown, and the final submission returns no score.

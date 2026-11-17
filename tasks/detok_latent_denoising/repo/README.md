# Robust Image Tokenizer

Supply the removed latent-corruption and missing-token behaviour of the l-DeTok image tokenizer so
that, after a fixed short continuation, its reconstructions are better (higher PSNR, lower LPIPS)
and the complete training-and-evaluation process is faster than the reference's. A valid
submission keeps the public tokenizer contract and runs the authentic training path. A
beats-reference criterion is evaluated alongside the score (see Scoring).

## Task

This repository trains an image tokenizer that turns images into latent tokens and reconstructs images from them. It reads fixed, licensed OpenImages V6 RGB photographs through the authors' list-based loader, using 64 training and 64 held-out validation images per workload. It produces ordinary-model and exponential-moving-average reconstructions; implementations must preserve tensor shapes, model parameter names and finite outputs. The objective rewards reconstruction quality after a fixed short continuation while also rewarding lower complete-process wall time. Several training-time corruption and missing-token behaviors in the tokenizer have been removed and must be supplied.

## What you may change

- `models/detok.py`, `models/autoencoder.py`
- new or changed modules below `models/frontier_solution/`, used through the two modules above.

At most 20 files may change; everything else is fixed. Constructors and public tokenizer APIs
stay fixed, as do returned tensor ranks, batch order, device and dtype compatibility and
state-dict parameter names (the retained checkpoint loader stops on a pretrained parameter it
cannot place).
The original decoder must receive a complete latent sequence in its spatial order. Helpers take
part only through the two editable modules; standalone programs or replacement training loops are
not evaluated. Open: corruption policies, token selection, missing-position representations,
sequence restoration and the training/evaluation balance.

## Interface

Preserve the existing call contracts in the two editable modules. `Encoder.mae_random_masking(x, mask_ratio)` accepts `[B,L,C]` tokens and returns visible tokens, a length-`L` binary mask, restoration indices (or `None` when nothing is removed), and aligned positional features. `Encoder.forward` carries that bookkeeping to `DeTok.encode`; `Decoder.forward` must restore a full length-`L` sequence before positional encoding. `DeTok.forward(x)` must return a reconstruction compatible with the unchanged reconstruction loss.

For example, an input `x` of shape `[2, 256, 768]` with a policy retaining 77 tokens yields visible tokens shaped `[2, 77, 768]`, a mask shaped `[2, 256]`, restoration indices shaped `[2, 256]`, and positional features whose second dimension is 77. Passing the resulting encoded sequence through the decoder must restore the decoder's expected 256 token positions and produce the same public image-shaped output contract as the unmodified model.

## Correctness

Score 0 when your method process (on any workload):
- crashes or exits non-zero (the stripped starter raises `NotImplementedError`);
- does not finish within 1,500 s;
- does not complete both the ordinary and the EMA evaluation, or leaves the evaluation summary or
  a reconstruction PNG missing or of the wrong shape;
- yields a non-finite measurement (an exact pixel match, with infinite PSNR, counts as one).

Metrics are recomputed by the judge from the saved PNGs.

## Workload

Each workload is an independently seeded fixed set of 64 training and 64 validation photographs.
One epoch with batch size 4: exactly 16 AdamW updates at learning rate `1e-6`, 256-pixel RGB
center crops, `gamma=3`, `mask_ratio=0.7` and the upstream inactive-at-epoch-zero discriminator
schedule; then the ordinary and EMA models are evaluated. In final the EMA evaluation uses a
second held-out set of 64 images; in train both use the same 64.

| suite | workloads |
|---|---|
| train | 3 |
| final | 3 |

The suites have disjoint membership and seeds and the same distribution. The judge runs the
retained `main_reconstruction.py` directly with `--epochs 1 --batch_size 4 --eval_bsz 4 --model
detok_BB --img_size 256 --patch_size 16 --token_channels 16 --gamma 3 --mask_ratio 0.7 --lr 1e-6
--discriminator_start_epoch 100 --num_workers 4 --class_of_interest 0 --num_images 64
--keep_eval_folder --save_freq 1 --vis_freq 100 --eval_freq 100` plus private data, checkpoint,
seed, FID-statistics and output paths. Timed is each side's complete method process; one untimed
warm-up (the reference on the first workload) runs first. FID and Inception Score are unscored.

## Scoring

Per workload, with `dO`, `dE` your ordinary and EMA quantized-PNG PSNR minus the reference's (dB),
`LO_r, LO_c` and `LE_r, LE_c` the reference's and your ordinary and EMA quantized-PNG LPIPS (v0.1,
VGG16, mean over the validation images) and `T_r, T_c` the complete method-process wall times:

```text
g       = [dO / 1.25, dE / 1.25, log2(max(LO_r, 1e-6) / max(LO_c, 1e-6)) / 0.25,
           log2(max(LE_r, 1e-6) / max(LE_c, 1e-6)) / 0.25]
speed   = 15 * tanh(log2(T_r / T_c) / 0.5)
if speed > 0: speed *= clamp((0.096 + min(g)) / 0.08, 0, 1)
score_w = 50 + 8.75 * (tanh(g[0]) + tanh(g[1]) + tanh(g[2]) + tanh(g[3])) + speed
score   = clamp(mean over workloads of score_w, 0, 100)
```

Anchors: the reference scores 50; quality gains alone reach at most 85, speed adds at most 15.

**Beats the reference** when, on every final workload, ordinary and EMA PSNR are strictly higher,
ordinary and EMA LPIPS strictly lower and the method-process wall time strictly lower than the
paired reference's.

## Running locally

On a GPU runner: `bash frontier-build.sh` compiles the tree; `bash frontier-bench.sh` runs the
retained entrypoint with `FRONTIER_DETOK_DATA`, `FRONTIER_DETOK_FID_STATS` and
`FRONTIER_DETOK_CHECKPOINT` set to your own inputs.

## Environment

CUDA 12.4, Python 3.10, PyTorch 2.6; one NVIDIA A100 80GB; no network on the GPU runner, so only
the installed packages and public assets are available.

## Notes

The workloads are representative, not images to recognise.

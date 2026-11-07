#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
test -n "${FRONTIER_DETOK_DATA:-}"
test -n "${FRONTIER_DETOK_FID_STATS:-}"
test -n "${FRONTIER_DETOK_CHECKPOINT:-}"
python main_reconstruction.py \
  --epochs 1 --batch_size 4 --eval_bsz 4 --model detok_BB \
  --img_size 256 --patch_size 16 --token_channels 16 \
  --gamma 3 --mask_ratio 0.7 --lr 1e-6 --seed "${FRONTIER_DETOK_SEED:-1}" \
  --discriminator_start_epoch 100 --num_workers 4 --num_images 64 \
  --data_path "$FRONTIER_DETOK_DATA/train" \
  --fid_stats_path "$FRONTIER_DETOK_FID_STATS" \
  --load_from "$FRONTIER_DETOK_CHECKPOINT" --keep_eval_folder \
  --output_dir "${FRONTIER_DETOK_OUTPUT:-/workspace/output}" \
  --save_freq 1 --vis_freq 100 --eval_freq 100

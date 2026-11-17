#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/frontier_runtime:$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python3 src/main.py \
  experiment=evaluate_only \
  dataset=dino_wm/point_maze \
  model=nanowm_b2 \
  experiment.resume_from_checkpoint="${NANOWM_CHECKPOINT:?set NANOWM_CHECKPOINT}" \
  dataset.loader.validation_fixed_subset_path="${NANOWM_SUBSET:?set NANOWM_SUBSET}" \
  experiment.evaluation.batch_size=1 \
  experiment.infra.num_workers=0 \
  experiment.infra.compile=false \
  wandb.enabled=false \
  logger.name=none

#!/bin/bash

DATA_DIR="data"
DATA_NAME="failed_test"

output_dir="${DATA_DIR}/large-language-models"
model_name="Qwen/Qwen3-8"

task="simple" # simple, five_shot
dataset_path="${DATA_DIR}/patterns/${DATA_NAME}.txt"
python scripts/inference.py \
  --task "${task}" \
  --dataset-path "${dataset_path}" \
  --output-dir "${output_dir}" \
  --tag "${DATA_NAME}" \
  --model-name "${model_name}"

# Note `--reasoning` argument
python scripts/inference.py \
  --task "${task}" \
  --dataset-path "${dataset_path}" \
  --output-dir "${output_dir}" \
  --tag "${DATA_NAME}" \
  --reasoning \
  --model-name "${model_name}"

task="localization" # localization, five_shot_localization
dataset_path="${DATA_DIR}/patterns/${DATA_NAME}.txt"
python scripts/inference.py \
  --task "${task}" \
  --dataset-path "${dataset_path}" \
  --output-dir "${output_dir}" \
  --tag "${DATA_NAME}" \
  --model-name "${model_name}"

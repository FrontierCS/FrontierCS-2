#!/usr/bin/env bash
set -euo pipefail
cd "${FRONTIER_REPO_ROOT:-/workspace/repo}"
MODEL_ROOT="${FRONTIER_MODEL_ROOT:-/opt/frontier/models}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=0
python train.py --config-name _2_sokoban \
  model_path="$MODEL_ROOT/qwen2.5-0.5b-instruct" \
  trainer.total_training_steps=1 \
  trainer.validation_steps=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  'trainer.logger=[console]' \
  trainer.generations_to_log_to_wandb.train=0 \
  trainer.generations_to_log_to_wandb.val=0 \
  es_manager.train.env_groups=6 \
  es_manager.train.group_size=4 \
  'es_manager.train.env_configs.n_groups=[6]' \
  actor_rollout_ref.rollout.response_length=96 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.actor.use_ref=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  ppo_mini_batch_size=4 \
  micro_batch_size_per_gpu=4 \
  agent_proxy.max_turn=3

#!/usr/bin/env bash
set -x
set -euo pipefail

# Backend for vLLM attention (override if needed)
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# ------------------------ Configurable Parameters ------------------------
# MUST be set by user (recommended: pass as env vars when launching)
# Example:
#   TRAIN_DATA=/abs/train.parquet VAL_DATA=/abs/val.parquet \
#   SFT_MERGED_CKPT=Qwen/Qwen2.5-3B-Instruct \
#   REWARD_FN=/abs/compute_score.py \
#   PROJECT_NAME=myproj EXPERIMENT_NAME=exp1 \
#   bash examples/grpo_trainer/med_v1_grpo.sh

# Parquet data paths
TRAIN_DATA=${TRAIN_DATA:-/path/to/train.parquet}
VAL_DATA=${VAL_DATA:-/path/to/val.parquet}

# Model checkpoint (local path or HF repo id)
SFT_MERGED_CKPT=${SFT_MERGED_CKPT:-/path/to/Qwen2.5-3B-Instruct_or_hf_repo_id}

# Custom reward function path
REWARD_FN=${REWARD_FN:-/path/to/compute_score.py}

# Logging / tracking
PROJECT_NAME=${PROJECT_NAME:-your_project_name}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-your_experiment_name}
LOG_FILE=${LOG_FILE:-${EXPERIMENT_NAME}.log}
# ------------------------------------------------------------------------

# Fail fast if user forgot to configure (placeholders still present)
if [[ "${TRAIN_DATA}" == "/path/to/train.parquet" ]] || [[ "${VAL_DATA}" == "/path/to/val.parquet" ]]; then
  echo "[ERROR] Please set TRAIN_DATA and VAL_DATA (placeholders detected)." >&2
  exit 1
fi
if [[ "${SFT_MERGED_CKPT}" == "/path/to/Qwen2.5-3B-Instruct_or_hf_repo_id" ]]; then
  echo "[ERROR] Please set SFT_MERGED_CKPT (placeholder detected)." >&2
  exit 1
fi
if [[ "${REWARD_FN}" == "/path/to/compute_score.py" ]]; then
  echo "[ERROR] Please set REWARD_FN (placeholder detected)." >&2
  exit 1
fi
if [[ "${PROJECT_NAME}" == "your_project_name" ]] || [[ "${EXPERIMENT_NAME}" == "your_experiment_name" ]]; then
  echo "[ERROR] Please set PROJECT_NAME and EXPERIMENT_NAME (placeholders detected)." >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size=480 \
  data.max_prompt_length=3072 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  critic.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.path="${SFT_MERGED_CKPT}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=96 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  reward_model.enable=false \
  +reward.reward_manager=custom \
  reward_model.micro_batch_size_per_gpu=1 \
  custom_reward_function.path="${REWARD_FN}" \
  trainer.critic_warmup=0 \
  trainer.default_hdfs_dir=null \
  trainer.logger=['console','wandb'] \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=-1 \
  trainer.total_epochs=3 \
  2>&1 | tee "${LOG_FILE}"

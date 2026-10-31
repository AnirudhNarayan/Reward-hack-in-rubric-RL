#!/usr/bin/env bash
# Launch one GRPO run with verl (called from scripts/slurm/train_grpo.slurm after Ray is up).
# Required env: DOMAIN (medical|science), VERIFIER (gpt-4o-mini|gpt-oss-120b), DATA_DIR, CKPT_ROOT
# Optional env: POLICY, NNODES, GPUS_PER_NODE, MAX_STEPS, EXTRA_ARGS
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

: "${DOMAIN:?}" "${VERIFIER:?}" "${DATA_DIR:?}" "${CKPT_ROOT:?}"
POLICY="${POLICY:-meta-llama/Llama-3.1-8B-Instruct}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RUN_NAME="${RUN_NAME:-${DOMAIN}_${VERIFIER}_$(basename "$POLICY")}"
CKPT_DIR="$CKPT_ROOT/$RUN_NAME"
mkdir -p "$CKPT_DIR"
export RRH_REWARD_LOG="$CKPT_DIR/reward_log.jsonl"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

STEP_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then STEP_ARGS+=(trainer.total_training_steps="$MAX_STEPS"); fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_DIR/$DOMAIN/train.parquet" \
  data.val_files="$DATA_DIR/$DOMAIN/eval300.parquet" \
  data.train_batch_size=128 \
  data.max_prompt_length=584 \
  data.max_response_length=2000 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$POLICY" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=4.2e-6 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$REPO/src/rrh/training/reward_fn.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.verifier="$VERIFIER" \
  reward.num_workers="${REWARD_WORKERS:-32}" \
  trainer.project_name=rubric-rh \
  trainer.experiment_name="$RUN_NAME" \
  trainer.logger='["console","wandb"]' \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.total_epochs=5 \
  trainer.save_freq=25 \
  trainer.test_freq=-1 \
  trainer.max_actor_ckpt_to_keep=null \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=auto \
  "${STEP_ARGS[@]}" ${EXTRA_ARGS:-}

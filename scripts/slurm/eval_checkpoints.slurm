#!/bin/bash
# GPU-side evaluation of every saved checkpoint of one run (+ step 0 = base policy):
#   1) merge verl FSDP shards -> HF
#   2) prompt-only generations on eval300 (exploitation rate, rewards, Sec. 4 analyses)
#   3) self-internalization gap raw scores (K=10 rubric-conditioned samples) + App. G.1 samples
#   4) medical only, steps % 50 == 0: HealthBench-1000 generations (App. F)
# API grading happens afterwards in scripts/slurm/grade_run.slurm (no GPU needed).
#   RUN_NAME=medical_gpt-4o-mini_Llama-3.1-8B-Instruct DOMAIN=medical \
#     sbatch -p <gpu-partition> -A <account> --gres=gpu:1 scripts/slurm/eval_checkpoints.slurm
#SBATCH --job-name=rrh-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
: "${RUN_NAME:?}" "${DOMAIN:?}"
POLICY="${POLICY:-meta-llama/Llama-3.1-8B-Instruct}"
CKPT_DIR="$CKPT_ROOT/$RUN_NAME"
OUT="$RUNS_DIR/$RUN_NAME"
EVAL="$DATA_DIR/$DOMAIN/eval300.jsonl"
HB="$DATA_DIR/healthbench/hb1000.jsonl"
STEPS="${STEPS:-}"   # optional explicit list, e.g. "0 25 50"
mkdir -p "$OUT"/{gen,selfgap,gen_rubric,healthbench_gen} "$SCRATCH_ROOT/hf_ckpts/$RUN_NAME"

if [[ -z "$STEPS" ]]; then
  STEPS="0 $(ls "$CKPT_DIR" | sed -n 's/^global_step_\([0-9]*\)$/\1/p' | sort -n | tr '\n' ' ')"
fi

for step in $STEPS; do
  if [[ "$step" == 0 ]]; then
    model="$POLICY"
  else
    model="$SCRATCH_ROOT/hf_ckpts/$RUN_NAME/step_$step"
    if [[ ! -f "$model/config.json" ]]; then
      python -m verl.model_merger merge --backend fsdp \
        --local_dir "$CKPT_DIR/global_step_$step/actor" --target_dir "$model"
    fi
  fi
  [[ -f "$OUT/gen/step$step.jsonl" ]] || python -m rrh.eval.generate_cli --model_path "$model" \
      --data "$EVAL" --step "$step" --context prompt --n 1 --out "$OUT/gen/step$step.jsonl"
  [[ -f "$OUT/selfgap/step$step.jsonl" ]] || python -m rrh.eval.self_gap_cli --model_path "$model" \
      --data "$EVAL" --step "$step" --K 10 --out "$OUT/selfgap/step$step.jsonl" \
      --samples_out "$OUT/gen_rubric/step$step.jsonl"
  # Sec. 4 set (500 prompts) only at the checkpoints listed in SEC4_STEPS (paper: 8 incl. base+last)
  if [[ " ${SEC4_STEPS:-} " == *" $step "* ]]; then
    mkdir -p "$OUT/gen_sec4"
    [[ -f "$OUT/gen_sec4/step$step.jsonl" ]] || python -m rrh.eval.generate_cli --model_path "$model" \
        --data "$DATA_DIR/$DOMAIN/sec4_500.jsonl" --step "$step" --context prompt --n 1 \
        --out "$OUT/gen_sec4/step$step.jsonl"
  fi
  if [[ "$DOMAIN" == medical && $((step % 50)) -eq 0 && -f "$HB" ]]; then
    [[ -f "$OUT/healthbench_gen/step$step.jsonl" ]] || python -m rrh.eval.generate_cli \
        --model_path "$model" --data "$HB" --step "$step" --context prompt --n 1 \
        --out "$OUT/healthbench_gen/step$step.jsonl"
  fi
done

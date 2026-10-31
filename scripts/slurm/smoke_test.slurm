#!/bin/bash
# Smoke test of the whole training path on ONE node: small Llama policy, weak (API) verifier,
# 50 iterations, checkpoints every 25. Verifies verl + reward fn + checkpointing before the
# expensive runs.
#   sbatch -p <gpu-partition> -A <account> --gres=gpu:<N> scripts/slurm/smoke_test.slurm
#SBATCH --job-name=rrh-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR}"
source "$REPO/scripts/env.sh"
export DOMAIN="${DOMAIN:-medical}" VERIFIER="${VERIFIER:-gpt-4o-mini}"
export POLICY="${POLICY:-meta-llama/Llama-3.2-1B-Instruct}"
export NNODES=1 GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
export MAX_STEPS="${MAX_STEPS:-50}" RUN_NAME="${RUN_NAME:-smoke_${DOMAIN}}"
ray start --head --num-gpus="$GPUS_PER_NODE"
bash "$REPO/scripts/train_grpo.sh"
ray stop

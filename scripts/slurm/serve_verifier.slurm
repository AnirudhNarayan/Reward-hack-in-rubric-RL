#!/bin/bash
# Serve a self-hosted judge (default GPT-OSS-120B, the strong verifier) with vLLM's
# OpenAI-compatible server and publish its URL for the training/grading jobs.
#   sbatch -p <gpu-partition> -A <account> --gres=gpu:2 scripts/slurm/serve_verifier.slurm
#   MODEL=openai/gpt-oss-20b URL_FILE=$SCRATCH_ROOT/verifier20b_url.txt sbatch ... (Table 4 row)
# Throughput note: 16 rollouts x 128 prompts = 2,048 grading calls per iteration; scale DP
# (more replicas behind separate jobs) if reward time dominates the step.
#SBATCH --job-name=rrh-verifier
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
URL_FILE="${URL_FILE:-$SCRATCH_ROOT/verifier_url.txt}"

echo "http://$(hostname --ip-address | awk '{print $1}'):$PORT/v1" > "$URL_FILE"
vllm serve "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" \
     --gpu-memory-utilization 0.90 --max-model-len 32768 --enable-prefix-caching

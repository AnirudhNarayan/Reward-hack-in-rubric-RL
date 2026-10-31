#!/bin/bash
# Long-running standalone judge server, for interactive use or sharing one server across jobs.
# The grade jobs start their own server, so the pipeline does not need this script.
# It writes the server URL to $URL_FILE; clients run: export JUDGE_BASE_URL=$(cat "$URL_FILE")
#   JUDGE=llama-3.3-70b-judge sbatch -p "$GPU_PARTITION" --gres=gpu:4 scripts/slurm/opd/serve_judge.slurm
#SBATCH --job-name=opd-judge-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
source "$REPO/scripts/slurm/opd/lib/vllm.sh"
: "${JUDGE:?}"
URL_FILE="${URL_FILE:-$SCRATCH_ROOT/judge_${JUDGE}_url.txt}"
start_vllm "$(model_field "$JUDGE" model)" "${JUDGE_TP:-$(nvidia-smi -L | wc -l)}" "${JUDGE_MAXLEN:-16384}"
echo "$JUDGE_BASE_URL" > "$URL_FILE"
echo "[serve] $JUDGE at $JUDGE_BASE_URL (written to $URL_FILE)"
wait "$VLLM_PID"

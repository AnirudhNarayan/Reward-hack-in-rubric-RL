#!/bin/bash
# One RP-OPD training segment (multi-node, multi-GPU, Newton). It resumes from the latest complete
# DCP checkpoint of $RUN_ID and exits cleanly about train.exit_margin_seconds before the walltime,
# so segments can be chained with --dependency=afterany (scripts/submit_opd_pipeline.sh does this).
# A segment that finds training already finished exits right away.
#   Required env: EXPERIMENT RUN_ID CONFIGS ; optional: OVERRIDES
#   Nodes, partition, GRES and time come from the submit script.
#SBATCH --job-name=opd-train
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
: "${CONFIGS:?}" "${RUN_ID:?}" "${EXPERIMENT:?}"
DEADLINE="$(job_deadline)"
NNODES="${SLURM_NNODES:-1}"
HEAD="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -1)"
export MASTER_ADDR="$HEAD"
export MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0      # colocated vLLM engine lives in the trainer process
echo "[train] nodes=$NNODES gpus/node=$GPUS_PER_NODE head=$HEAD deadline=${DEADLINE:-none} run=$RUN_ID"

EXTRA=()
[[ -n "$DEADLINE" ]] && EXTRA+=(--deadline "$DEADLINE")
# shellcheck disable=SC2086
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
  torchrun --nnodes "$NNODES" --nproc_per_node "$GPUS_PER_NODE" \
           --rdzv_id "${SLURM_JOB_ID:-0}" --rdzv_backend c10d --rdzv_endpoint "$HEAD:$MASTER_PORT" \
  -m rrh.opd.train_cli --config $CONFIGS --run_id "$RUN_ID" --results_root "$RESULTS_ROOT" \
     "${EXTRA[@]}" --set ${OVERRIDES:-}

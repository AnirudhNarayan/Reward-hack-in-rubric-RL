#!/bin/bash
# Grade a run's generations with one judge. For self-hosted judges (provider: vllm) this job starts
# the vLLM server on its own GPUs and stops it at the end. API judges need no GPU.
#   Required env: EXPERIMENT RUN_ID JUDGE
#   TASKS: space-separated "<mode>:<subset_n|all>[:<step+step+...>]", for example
#     TASKS="boolean:all"
#     TASKS="shortcut:all boolean:200"
#     TASKS="boolean:200:step_000000+teacher"   (use + between steps: sbatch --export splits on commas)
#   optional: JUDGE_TP (default: every GPU in the job), JUDGE_MAXLEN
#SBATCH --job-name=opd-grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
source "$REPO/scripts/slurm/opd/lib/vllm.sh"
: "${JUDGE:?}"
TASKS="${TASKS:-boolean:all}"
RD="$(run_dir)"
export VLLM_LOG="$RD/logs/vllm-${JUDGE}-${SLURM_JOB_ID:-local}.log"
if [[ "$(model_field "$JUDGE" provider)" == vllm ]]; then
  start_vllm "$(model_field "$JUDGE" model)" "${JUDGE_TP:-$(nvidia-smi -L | wc -l)}" "${JUDGE_MAXLEN:-16384}"
fi
for t in $TASKS; do
  IFS=: read -r mode subset steps <<< "$t"
  args=(--run_dir "$RD" --judge "$JUDGE" --mode "$mode")
  if [[ -n "$subset" && "$subset" != all ]]; then args+=(--subset_n "$subset"); fi
  if [[ -n "${steps:-}" ]]; then args+=(--steps ${steps//+/ }); fi
  python -m rrh.grading.opd_grader "${args[@]}"
done

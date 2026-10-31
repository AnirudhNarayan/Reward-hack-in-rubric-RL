#!/bin/bash
# Judge-substitution validation (CPU): agreement / FP / FN against the strong judge, plus score gaps.
#   Required env: EXPERIMENT RUN_ID ; optional REFERENCE, JUDGES, OUT_DIR
#SBATCH --job-name=opd-judgeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
RD="$(run_dir)"
# shellcheck disable=SC2086
python -m rrh.analysis.judge_agreement --run_dirs "$RD" \
  --reference "${REFERENCE:-llama-3.3-70b-judge}" \
  --judges ${JUDGES:-gpt-oss-120b-ref gemma-3-27b-judge gemma-3-27b-judge-shortcut} \
  --out_dir "${OUT_DIR:-$RD}"

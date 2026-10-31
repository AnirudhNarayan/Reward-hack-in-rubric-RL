#!/bin/bash
# API grading for one run (CPU only; needs internet on the node, or run from a node that has it).
# Grades every prompt-only eval generation with the training verifier + the 3-judge panel,
# the App. G.1 rubric-conditioned samples (steps % 50 == 0) with the panel, and HealthBench.
#   RUN_NAME=... DOMAIN=medical VERIFIER=gpt-4o-mini sbatch -p <cpu-partition> scripts/slurm/grade_run.slurm
#SBATCH --job-name=rrh-grade
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
: "${RUN_NAME:?}" "${DOMAIN:?}" "${VERIFIER:?}"
PANEL="${PANEL:-gpt-5.4,gemini-3-pro,claude-opus-4.6}"
OUT="$RUNS_DIR/$RUN_NAME"
EVAL="$DATA_DIR/$DOMAIN/eval300.jsonl"
if [[ "${VERIFIER}" == gpt-oss* ]]; then
  export VERIFIER_BASE_URL="$(cat "${VERIFIER_URL_FILE:-$SCRATCH_ROOT/verifier_url.txt}")"
fi

for f in "$OUT"/gen/step*.jsonl; do
  python -m rrh.grading.grade_cli --generations "$f" --data "$EVAL" \
      --judges "$VERIFIER,$PANEL" --out "$OUT/judgments.jsonl"
done
if [[ "${SKIP_G1:-0}" != 1 ]]; then
  for f in "$OUT"/gen_rubric/step*.jsonl; do
    step=$(basename "$f" .jsonl); step=${step#step}
    (( step % 50 == 0 )) || continue
    python -m rrh.grading.grade_cli --generations "$f" --data "$EVAL" \
        --judges "$PANEL" --out "$OUT/judgments_rubric_cond.jsonl"
  done
fi
if [[ "$DOMAIN" == medical ]]; then
  for f in "$OUT"/healthbench_gen/step*.jsonl; do
    python -m rrh.eval.healthbench_grade --generations "$f" \
        --data "$DATA_DIR/healthbench/hb1000.jsonl" --out "$OUT/healthbench_scores.jsonl"
  done
fi

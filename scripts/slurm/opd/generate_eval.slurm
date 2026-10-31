#!/bin/bash
# Generate eval responses (App. A.3 decoding) for every complete HF checkpoint of a run.
# Each checkpoint runs in its own python process so GPU memory is released cleanly. Files that
# were already generated are skipped, which makes the job resumable.
#   Required env: EXPERIMENT RUN_ID ; optional: STEPS="0 100 980", TP=1
#SBATCH --job-name=opd-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
RD="$(run_dir)"
if [[ -z "${STEPS:-}" ]]; then
  STEPS="$(python -m rrh.opd.list_checkpoints "$RD")"
fi
echo "[gen] run=$RD steps: $STEPS"
for s in $STEPS; do
  python -m rrh.opd.evaluate --run_dir "$RD" --steps "$s" --tp "${TP:-1}"
done

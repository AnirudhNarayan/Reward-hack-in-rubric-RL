#!/bin/bash
# Rebuild every figure and table, plus the INDEX.csv metrics, from results/ (CPU).
#SBATCH --job-name=opd-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
python -m rrh.analysis.opd_report --results_root "$RESULTS_ROOT"

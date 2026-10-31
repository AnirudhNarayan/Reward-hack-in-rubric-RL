#!/bin/bash
# Phase 0: download RubricHub + HealthBench and build splits (CPU; needs internet).
#   sbatch -p <cpu-partition> scripts/slurm/prepare_data.slurm
#SBATCH --job-name=rrh-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
python -m rrh.data.rubrichub --out_dir "$DATA_DIR" ${SOURCE_FILTER:+--source_filter}
python -m rrh.data.healthbench --out "$DATA_DIR/healthbench/hb1000.jsonl"

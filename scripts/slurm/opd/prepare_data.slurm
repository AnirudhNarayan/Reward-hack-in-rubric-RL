#!/bin/bash
# Data prep (CPU partition; Stokes is fine if it shares the filesystem) + model download to $HF_HOME.
# Needs internet for Hugging Face. If compute nodes are offline (COMPUTE_HAS_INTERNET=0), run it
# on a node that has internet access.
#   Submitted by scripts/submit_opd_pipeline.sh, or on its own:
#   sbatch -p "$CPU_PARTITION" scripts/slurm/opd/prepare_data.slurm
#SBATCH --job-name=opd-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
DATASETS="${DATASETS:-healthbench researchqa rubrichub_science}"
# Training models only. Each grade job downloads its own judge when it starts (D16: judges arrive
# after the intermediate checkpoints are gone, which keeps peak disk under the 1 TB quota).
MODELS="${MODELS:-meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct}"

if [[ ! -f "$DATA_DIR/opd/manifest.json" || -n "${FORCE_DATA:-}" ]]; then
  # shellcheck disable=SC2086
  python -m rrh.data.opd_data --out_dir "$DATA_DIR/opd" --only $DATASETS --cache_dir "$HF_HOME/hub"
fi
cat "$DATA_DIR/opd/manifest.json"

# shellcheck disable=SC2086
python -m rrh.data.fetch_models $MODELS

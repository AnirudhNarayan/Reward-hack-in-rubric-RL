#!/bin/bash
# Fig. 2 gold line: rubric-conditioned teacher (Llama-3.1-70B) generations on the eval sets, using
# the same decoding as the student. Needs TP GPUs (70B in bf16: TP=4 on 80 GB GPUs, 8 on 40 GB).
#   Required env: EXPERIMENT RUN_ID ; optional TP
#SBATCH --job-name=opd-teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
python -m rrh.opd.evaluate --run_dir "$(run_dir)" --policy teacher --tp "${TP:-$(nvidia-smi -L | wc -l)}"

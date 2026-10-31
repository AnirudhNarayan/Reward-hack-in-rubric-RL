# Copy to scripts/env.sh (git-ignored) and edit for Stokes / Newton.
# Sourced by every SLURM script. Never put API keys here: they live in ~/.env.

# module load cuda/12.4 anaconda   # <- cluster modules
# conda activate rrh

export SCRATCH_ROOT="${SCRATCH_ROOT:-/lustre/fs1/home/$USER/rrh}"   # EDIT: scratch, not $HOME
export DATA_DIR="$SCRATCH_ROOT/data"
export CKPT_ROOT="$SCRATCH_ROOT/checkpoints"
export RUNS_DIR="$SCRATCH_ROOT/runs"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/src:${PYTHONPATH:-}"
mkdir -p "$DATA_DIR" "$CKPT_ROOT" "$RUNS_DIR" logs

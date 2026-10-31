#!/bin/bash
# Multi-node GRPO under SLURM + Ray. Submit (partition/account/gres are cluster-specific):
#   DOMAIN=medical VERIFIER=gpt-4o-mini sbatch -p <gpu-partition> -A <account> \
#       --gres=gpu:8 scripts/slurm/train_grpo.slurm
# For VERIFIER=gpt-oss-120b, first start scripts/slurm/serve_verifier.slurm and wait until it
# writes $VERIFIER_URL_FILE (default $SCRATCH_ROOT/verifier_url.txt).
#SBATCH --job-name=rrh-grpo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR}"
source "$REPO/scripts/env.sh"

if [[ "${VERIFIER}" == gpt-oss* ]]; then
  export VERIFIER_BASE_URL="$(cat "${VERIFIER_URL_FILE:-$SCRATCH_ROOT/verifier_url.txt}")"
  echo "verifier at $VERIFIER_BASE_URL"
fi

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname --ip-address | awk '{print $1}')
port=6379
export RAY_ADDRESS="$head_ip:$port"

srun --nodes=1 --ntasks=1 -w "$head" ray start --head --node-ip-address="$head_ip" --port=$port \
     --num-gpus="$GPUS_PER_NODE" --block &
sleep 20
for n in "${nodes[@]:1}"; do
  srun --nodes=1 --ntasks=1 -w "$n" ray start --address="$RAY_ADDRESS" \
       --num-gpus="$GPUS_PER_NODE" --block &
done
sleep 20

NNODES=$SLURM_NNODES srun --overlap --nodes=1 --ntasks=1 -w "$head" bash "$REPO/scripts/train_grpo.sh"

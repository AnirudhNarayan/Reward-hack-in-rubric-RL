#!/usr/bin/env bash
# Inspect a SLURM cluster and write docs/cluster/<name>.raw.txt + a draft scripts/cluster.<name>.env.
# Run ONCE on each login node (Newton, Stokes). The only compute it does is a 5-minute `srun` per
# GPU partition (nvidia-smi, CPU/mem, and an internet probe from a compute node).
#
#   bash scripts/cluster/inspect_cluster.sh newton
#   bash scripts/cluster/inspect_cluster.sh stokes
#   SKIP_SRUN=1 bash scripts/cluster/inspect_cluster.sh stokes     # metadata only
#
# Then copy the confirmed values into docs/CLUSTER.md and scripts/cluster.env.
set -uo pipefail
NAME="${1:?usage: inspect_cluster.sh <cluster-name>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$REPO/docs/cluster/$NAME.raw.txt"
mkdir -p "$REPO/docs/cluster"
exec > >(tee "$OUT") 2>&1

sec() { printf '\n==================== %s ====================\n' "$*"; }

sec "host / date"
hostname; date; uname -a; echo "USER=$USER HOME=$HOME"
sec "sinfo summary"
sinfo -s
sec "sinfo per partition (partition|avail|timelimit|nodes|state|cpus|mem_MB|gres|features)"
sinfo -o "%P|%a|%l|%D|%t|%c|%m|%G|%f" | sort -u
sec "scontrol show partition"
scontrol show partition
sec "accounts / QOS for $USER"
sacctmgr -nP show assoc user="$USER" format=cluster,account,partition,qos,maxjobs,maxwall,grptres 2>/dev/null || echo "sacctmgr unavailable"
sacctmgr -nP show qos format=name,maxwall,maxtresperuser,maxjobsperuser 2>/dev/null | head -50
sec "module avail (cuda / python / conda / gcc / nccl)"
if command -v module >/dev/null 2>&1; then
  module -t avail 2>&1 | grep -i -E "cuda|anaconda|conda|miniconda|python|gcc|nccl|cudnn" | sort -u | head -100
else
  echo "no 'module' command"
fi
sec "storage"
df -h "$HOME" 2>/dev/null
for d in "/lustre/fs1/home/$USER" "/scratch/$USER" "/lustre/scratch/$USER" "${SCRATCH:-}" "${WORK:-}"; do
  [[ -n "$d" && -d "$d" ]] && { echo "--- $d"; df -h "$d"; }
done
command -v quota >/dev/null && quota -s 2>/dev/null
command -v lfs >/dev/null && lfs quota -h -u "$USER" /lustre/fs1 2>/dev/null
sec "login-node internet"
curl -s -o /dev/null -m 10 -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co || echo "huggingface.co unreachable"
curl -s -o /dev/null -m 10 -w "api.openai.com HTTP %{http_code}\n" https://api.openai.com/v1/models || echo "api.openai.com unreachable"

if [[ -z "${SKIP_SRUN:-}" ]]; then
  # GPU partitions = partitions whose GRES mentions gpu
  mapfile -t GPU_PARTS < <(sinfo -h -o "%P %G" | awk 'tolower($2) ~ /gpu/ {gsub(/\*/,"",$1); print $1}' | sort -u)
  for P in "${GPU_PARTS[@]}"; do
    sec "compute probe: partition $P (srun, 1 GPU, 5 min)"
    timeout 900 srun -p "$P" -N1 -n1 --gres=gpu:1 -c 4 --mem=16G -t 00:05:00 bash -lc '
      hostname; nvidia-smi; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv;
      nvidia-smi topo -m 2>/dev/null | head -20; echo "nproc=$(nproc)"; free -g | head -2;
      ls /dev/infiniband 2>/dev/null && echo "infiniband: yes" || echo "infiniband: not visible";
      curl -s -o /dev/null -m 10 -w "compute->huggingface.co HTTP %{http_code}\n" https://huggingface.co || echo "compute: no internet (huggingface)";
      curl -s -o /dev/null -m 10 -w "compute->api.openai.com HTTP %{http_code}\n" https://api.openai.com/v1/models || echo "compute: no internet (openai)";
      df -h /tmp | tail -1' || echo "srun on $P failed (account/QOS? try: sbatch with -A)"
  done
  CPU_PART="$(sinfo -h -o "%P %G" | awk 'tolower($2) !~ /gpu/ {gsub(/\*/,"",$1); print $1; exit}')"
  if [[ -n "$CPU_PART" ]]; then
    sec "compute probe: CPU partition $CPU_PART"
    timeout 600 srun -p "$CPU_PART" -N1 -n1 -c 2 --mem=4G -t 00:03:00 bash -lc '
      hostname; nproc; free -g | head -2;
      curl -s -o /dev/null -m 10 -w "compute->huggingface.co HTTP %{http_code}\n" https://huggingface.co || echo "compute: no internet"' \
      || echo "srun on $CPU_PART failed"
  fi
fi
sec "done"
echo "Wrote $OUT. Fill docs/CLUSTER.md and scripts/cluster.env from it (see scripts/cluster.env.example)."

#!/usr/bin/env bash
# One-time environment setup on the cluster (login node is fine: installs only, no compute).
#   bash scripts/setup_env.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-rrh}"

# Modules + scratch paths come from scripts/cluster.env (Newton: cp scripts/cluster.newton.env).
if [[ -f "$REPO/scripts/cluster.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/scripts/cluster.env"
  # shellcheck disable=SC2086
  command -v module >/dev/null 2>&1 && [[ -n "${MODULES:-}" ]] && module load $MODULES
  # keep the multi-GB pip cache off $HOME
  [[ -n "${SCRATCH_ROOT:-}" ]] && export PIP_CACHE_DIR="$SCRATCH_ROOT/pip_cache" && mkdir -p "$PIP_CACHE_DIR"
fi
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# verl pins its own torch / vllm; install it first, then our package on top.
pip install "verl[vllm]==0.9.1"
pip install flash-attn --no-build-isolation || echo "flash-attn build failed; verl falls back to sdpa"
pip install -e "$REPO[dev]" anthropic huggingface_hub

# Canonical HealthBench grader template (App. F).
mkdir -p "$REPO/third_party"
if [[ ! -d "$REPO/third_party/simple-evals" ]]; then
  git clone --depth 1 https://github.com/openai/simple-evals "$REPO/third_party/simple-evals"
fi

[[ -f "$REPO/scripts/env.sh" ]] || cp "$REPO/scripts/env.example.sh" "$REPO/scripts/env.sh"
[[ -f "$REPO/configs/local.yaml" ]] || cp "$REPO/configs/local.yaml.example" "$REPO/configs/local.yaml"
echo "Done. Edit scripts/env.sh and configs/local.yaml; put API keys in ~/.env (chmod 600)."

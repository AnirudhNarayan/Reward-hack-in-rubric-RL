#!/usr/bin/env bash
# Sourced by every scripts/slurm/opd/*.slurm job: cluster settings, modules, conda, paths.
set -euo pipefail
REPO="${RRH_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
cd "$REPO"
if [[ ! -f "$REPO/scripts/cluster.env" ]]; then
  echo "missing scripts/cluster.env (copy scripts/cluster.env.example and fill it in)" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$REPO/scripts/cluster.env"
# shellcheck disable=SC1091
[[ -f "$REPO/scripts/env.sh" ]] && source "$REPO/scripts/env.sh"

if command -v module >/dev/null 2>&1 && [[ -n "${MODULES:-}" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-rrh}"
fi

# HF_TOKEN for gated Llama/Gemma downloads by vLLM / huggingface_hub (never echoed). Other keys in
# ~/.env stay unexported; Python code reads them via rrh.utils.io.load_env.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN[[:space:]]*=[[:space:]]*//p' "$HOME/.env" \
              | tail -1 | tr -d "\"'\r")"
  [[ -n "$HF_TOKEN" ]] && export HF_TOKEN
fi

export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
export RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"
export HF_HOME DATA_DIR CKPT_ROOT RRH_CACHE
export TOKENIZERS_PARALLELISM=false
export RRH_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ "${COMPUTE_HAS_INTERNET:-1}" == 0 ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
mkdir -p "$DATA_DIR" "$CKPT_ROOT" "$HF_HOME" "$(dirname "$RRH_CACHE")"

# run directory of the current pipeline (set by scripts/submit_opd_pipeline.sh)
run_dir() { echo "$RESULTS_ROOT/${EXPERIMENT:?}/${RUN_ID:?}"; }

# unix time at which this job's walltime ends (empty if unknown)
job_deadline() {
  [[ -z "${SLURM_JOB_ID:-}" ]] && return 0
  local end
  end="$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)"
  [[ -n "$end" && "$end" != "N/A" ]] && date -d "$end" +%s 2>/dev/null || true
}

# python one-liner: field of a models.yaml alias (e.g. served model id)
model_field() {
  python - "$1" "$2" <<'EOF'
import sys, yaml
spec = yaml.safe_load(open("configs/models.yaml"))["models"][sys.argv[1]]
print(spec.get(sys.argv[2], ""))
EOF
}

echo "[common] host=$(hostname) job=${SLURM_JOB_ID:-none} repo=$REPO commit=${RRH_GIT_COMMIT:0:10}"

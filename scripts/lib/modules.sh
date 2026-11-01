#!/usr/bin/env bash
# Sourced by setup_env.sh and scripts/slurm/opd/lib/common.sh: load $MODULES, then make `conda` usable.
#
# Newton's anaconda Lmod module defines conda as shell functions. Bash functions do not reach child
# shells (sbatch jobs, `bash script.sh`) but the module's env vars do, so `module load` there sees a
# half-loaded conda and loops on "CondaError: Run 'conda init' before 'conda deactivate'".
# So: skip modules that are already loaded, skip the anaconda module when conda is already on hand,
# and get the conda functions from conda.sh instead.
load_cluster_modules() {
  if command -v module >/dev/null 2>&1 && [[ -n "${MODULES:-}" ]]; then
    local m
    for m in $MODULES; do
      module is-loaded "$m" >/dev/null 2>&1 && continue
      [[ "$m" == anaconda* && -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]] && continue
      module load "$m"
    done
  fi
  local conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
  if [[ -n "$conda_bin" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$(dirname "$conda_bin")")/etc/profile.d/conda.sh"
  fi
}

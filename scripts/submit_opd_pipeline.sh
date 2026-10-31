#!/usr/bin/env bash
# Submit the full RP-OPD pipeline for one dataset with SLURM dependencies (run on a login node;
# it only calls sbatch).
#
# Scope: stage 1 (RP-OPD) only. This pipeline never launches rubric-based RL / GRPO (stage 2).
#
#   bash scripts/submit_opd_pipeline.sh --dataset healthbench
#   bash scripts/submit_opd_pipeline.sh --dataset all --seed 0
#   bash scripts/submit_opd_pipeline.sh --dataset rubrichub_science --variant reverse_kl
#   bash scripts/submit_opd_pipeline.sh --smoke                 # 1-GPU smoke test only
#   bash scripts/submit_opd_pipeline.sh --dataset healthbench --dry-run
#
# DAG per dataset (prep is shared):
#   prep(CPU) -> train_1 -> train_2 ... train_S   (afterany chain; each segment resumes)
#   prep -> teacher_ref(GPU, TP)                   (independent of training)
#   train_S -> gen(GPU) ; gen + teacher_ref -> grade_curve (Llama-3.3-70B, all checkpoints)
#                                            -> grade_indep (Gemma-3-27B shortcut-aware, all;
#                                                            + A.1 boolean on a 200-prompt subset)
#   gen + teacher_ref -> grade_ref (gpt-oss-120b, 200-prompt subset)
#   grades -> judge_validation(CPU) -> report(CPU)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
[[ -f scripts/cluster.env ]] || { echo "create scripts/cluster.env first (see docs/CLUSTER.md)"; exit 2; }
# shellcheck disable=SC1091
source scripts/cluster.env

DATASET=healthbench; SEED=0; SEGMENTS=4; NODES="${TRAIN_NODES:-2}"; VARIANT=""; DRY=""; SMOKE=""
SKIP_PREP=""; SEG_TIME="${MAX_WALLTIME:-48:00:00}"; VAL_SUBSET=200; OVERRIDES="${OVERRIDES:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --segments) SEGMENTS="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --segment-time) SEG_TIME="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;          # rubric_blind | reverse_kl
    --skip-prep) SKIP_PREP=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

SB_COMMON=()
[[ -n "${ACCOUNT:-}" ]] && SB_COMMON+=(-A "$ACCOUNT")
[[ -n "${QOS:-}" ]] && SB_COMMON+=(--qos "$QOS")
gres() { echo "gpu:${GPU_TYPE:+$GPU_TYPE:}$1"; }
GPU_ARGS=(-p "${GPU_PARTITION:?set GPU_PARTITION in scripts/cluster.env}")
CPU_ARGS=(-p "${CPU_PARTITION:-$GPU_PARTITION}")

submit() {   # submit <name> <logdir> <sbatch args...> <script>; prints job id
  local name="$1" logdir="$2"; shift 2
  mkdir -p "$logdir"
  if [[ -n "$DRY" ]]; then
    echo "DRY sbatch --parsable ${SB_COMMON[*]} --output=$logdir/%x-%j.out $*" >&2
    echo "$((RANDOM + 100000))"
  else
    sbatch --parsable "${SB_COMMON[@]}" --output="$logdir/%x-%j.out" "$@"
  fi
}

LOGROOT="$REPO/results/_pipeline_logs"
if [[ -n "$SMOKE" ]]; then
  PREP=""
  if [[ -z "$SKIP_PREP" ]]; then
    PREP=$(submit prep "$LOGROOT" "${CPU_ARGS[@]}" --export=ALL,DATASETS=healthbench,MODELS="meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct" scripts/slurm/opd/prepare_data.slurm)
  fi
  J=$(submit smoke "$LOGROOT" "${GPU_ARGS[@]}" --gres="$(gres 1)" --cpus-per-task="${CPUS_PER_GPU:-8}" \
      ${PREP:+--dependency=afterok:$PREP} scripts/slurm/opd/smoke_test.slurm)
  echo "smoke: prep=${PREP:-skipped} smoke=$J   (logs: $LOGROOT)"
  exit 0
fi

PREP=""
if [[ -z "$SKIP_PREP" ]]; then
  PREP=$(submit prep "$LOGROOT" "${CPU_ARGS[@]}" scripts/slurm/opd/prepare_data.slurm)
  echo "prep: $PREP"
fi

run_one() {
  local ds="$1" ds_name exp cfgs run_id rd
  case "$ds" in
    healthbench) ds_name=healthbench ;;
    researchqa) ds_name=researchqa ;;
    rubrichub_science) ds_name=rubrichub-science ;;
    *) echo "unknown dataset $ds"; exit 2 ;;
  esac
  exp=rp_opd
  cfgs="configs/opd/base.yaml configs/opd/$ds.yaml"
  case "$VARIANT" in
    "") ;;
    rubric_blind) exp=opd_rubric_blind; cfgs="$cfgs configs/opd/rubric_blind.yaml" ;;
    reverse_kl) exp=rp_opd_reverse_kl; cfgs="$cfgs configs/opd/reverse_kl.yaml" ;;
    *) echo "unknown variant $VARIANT"; exit 2 ;;
  esac
  run_id="llama-3.1-8b-instruct__llama-3.3-70b-judge__${ds_name}__seed${SEED}__$(date +%Y%m%d-%H%M)"
  rd="$REPO/results/$exp/$run_id"
  local L="$rd/logs"
  local ENV="ALL,EXPERIMENT=$exp,RUN_ID=$run_id"
  local dep_prep=${PREP:+--dependency=afterok:$PREP}

  # training segments
  local prev="" seg ids=()
  for seg in $(seq 1 "$SEGMENTS"); do
    local dep="$dep_prep"
    [[ -n "$prev" ]] && dep="--dependency=afterany:$prev"
    prev=$(submit train "$L" "${GPU_ARGS[@]}" -N "$NODES" --gres="$(gres "$GPUS_PER_NODE")" \
           --cpus-per-task="$(( ${CPUS_PER_GPU:-8} * GPUS_PER_NODE ))" --mem="${MEM_PER_NODE:-0}" \
           --time="$SEG_TIME" $dep \
           --export="$ENV,CONFIGS=$cfgs,OVERRIDES=seed=$SEED $OVERRIDES" scripts/slurm/opd/train_opd.slurm)
    ids+=("$prev")
  done
  local TEACH GEN GC GI GR JV REP
  TEACH=$(submit teacher "$L" "${GPU_ARGS[@]}" --gres="$(gres 4)" --cpus-per-task=32 --mem=256G $dep_prep \
          --export="$ENV" scripts/slurm/opd/teacher_reference.slurm)
  GEN=$(submit gen "$L" "${GPU_ARGS[@]}" --gres="$(gres 1)" --cpus-per-task=8 --mem=64G \
        --dependency=afterany:"$prev" --export="$ENV" scripts/slurm/opd/generate_eval.slurm)
  GC=$(submit grade "$L" "${GPU_ARGS[@]}" --gres="$(gres 4)" --cpus-per-task=32 --mem=256G \
       --dependency=afterok:"$GEN":"$TEACH" --export="$ENV,JUDGE=llama-3.3-70b-judge,TASKS=boolean:all" \
       scripts/slurm/opd/grade.slurm)
  GI=$(submit grade "$L" "${GPU_ARGS[@]}" --gres="$(gres 2)" --cpus-per-task=16 --mem=128G \
       --dependency=afterok:"$GEN":"$TEACH" \
       --export="$ENV,JUDGE=gemma-3-27b-judge,TASKS=shortcut:all boolean:$VAL_SUBSET" scripts/slurm/opd/grade.slurm)
  GR=$(submit grade "$L" "${GPU_ARGS[@]}" --gres="$(gres 2)" --cpus-per-task=16 --mem=128G \
       --dependency=afterok:"$GEN":"$TEACH" \
       --export="$ENV,JUDGE=gpt-oss-120b-ref,TASKS=boolean:$VAL_SUBSET" scripts/slurm/opd/grade.slurm)
  JV=$(submit judgeval "$L" "${CPU_ARGS[@]}" --dependency=afterany:"$GC":"$GI":"$GR" \
       --export="$ENV" scripts/slurm/opd/judge_validation.slurm)
  REP=$(submit report "$L" "${CPU_ARGS[@]}" --dependency=afterany:"$JV" --export="$ENV" scripts/slurm/opd/report.slurm)
  if [[ -z "$DRY" ]]; then
    { echo "run_id=$run_id experiment=$exp dataset=$ds seed=$SEED submitted=$(date -Is)"
      echo "prep=${PREP:-skipped} train=${ids[*]} teacher=$TEACH gen=$GEN grade_curve=$GC grade_indep=$GI grade_ref=$GR judgeval=$JV report=$REP"
    } >> "$L/pipeline_jobs.txt"
  fi
  echo "[$ds] run=$rd"
  echo "  train=${ids[*]} teacher=$TEACH gen=$GEN grade: curve=$GC indep=$GI ref=$GR judgeval=$JV report=$REP"
}

if [[ "$DATASET" == all ]]; then
  for d in healthbench researchqa rubrichub_science; do run_one "$d"; done
else
  run_one "$DATASET"
fi

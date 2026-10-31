#!/bin/bash
# End-to-end smoke test on ONE GPU, run before any scale-up:
#   Llama-3.2-1B student, Llama-3.2-3B rubric-aware teacher, 50 HealthBench prompts, 4 OPD steps
#   -> HF checkpoints -> eval generations (50 prompts) -> teacher reference -> judge grading
#   (Llama-3.1-8B smoke judge, A.1 boolean + C.1 shortcut-aware) -> report -> INDEX.csv.
#   Needs the data from prepare_data.slurm. ROLLOUT=hf skips vLLM for the rollouts.
#   sbatch -p "$GPU_PARTITION" --gres=gpu:1 scripts/slurm/opd/smoke_test.slurm
#SBATCH --job-name=opd-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
source "$REPO/scripts/slurm/opd/lib/vllm.sh"
export EXPERIMENT=smoke
export RUN_ID="${RUN_ID:-llama-3.2-1b-instruct__llama-3.1-8b-smoke-judge__healthbench__seed0__$(date +%Y%m%d-%H%M)}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
RD="$(run_dir)"
mkdir -p "$RD/logs"
CONFIGS="configs/opd/base.yaml configs/opd/healthbench.yaml configs/opd/smoke.yaml"

echo "== 1/5 train (4 steps, rollout=${ROLLOUT:-vllm})"
# shellcheck disable=SC2086
python -m rrh.opd.train_cli --config $CONFIGS --run_id "$RUN_ID" --results_root "$RESULTS_ROOT" \
  --set "rollout.backend=${ROLLOUT:-vllm}" "student.attn_implementation=sdpa" "teacher.attn_implementation=sdpa"

echo "== 2/5 eval generations (student checkpoints) + teacher reference"
for s in $(python -m rrh.opd.list_checkpoints "$RD"); do
  python -m rrh.opd.evaluate --run_dir "$RD" --steps "$s"
done
python -m rrh.opd.evaluate --run_dir "$RD" --policy teacher --tp 1

echo "== 3/5 judge (vLLM on the same GPU)"
export VLLM_LOG="$RD/logs/vllm-smoke-judge.log"
start_vllm "$(model_field llama-3.1-8b-smoke-judge model)" 1 16384
python -m rrh.grading.opd_grader --run_dir "$RD" --judge llama-3.1-8b-smoke-judge --mode boolean
python -m rrh.grading.opd_grader --run_dir "$RD" --judge llama-3.1-8b-smoke-judge --mode shortcut
stop_vllm

echo "== 4/5 report"
python -m rrh.analysis.opd_report --results_root "$RESULTS_ROOT" --experiments smoke

echo "== 5/5 checks"
python - "$RD" <<'EOF'
import json, sys
from pathlib import Path
from rrh.results.layout import RunDir, read_index
rd = RunDir(sys.argv[1])
steps = [json.loads(l)["step"] for l in open(rd.train_metrics())]
assert steps == [1, 2, 3, 4], steps
summ = json.loads((rd.path / "metrics" / "summary.json").read_text())
assert "heldout" in summ, summ
assert (rd.path / "figures" / "fig2_heldout.png").exists()
row = [r for r in read_index(rd.results_root) if r["run_id"] == rd.run_id][0]
print("SMOKE OK", json.dumps(summ["heldout"]), "status:", row["status"])
EOF

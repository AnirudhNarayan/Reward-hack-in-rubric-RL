#!/bin/bash
# Multi-GPU smoke test on ONE node with 2 GPUs, run after smoke_test.slurm and before the full run.
# It covers what the 1-GPU test cannot (FSDP2 and vLLM external_launcher only run with >1 rank):
#   1. FSDP2 loading: sharded Llama-3.2-1B vs a plain load give the same next-token log-probs
#   2. 2-rank training (FSDP2 student + teacher, vLLM rollouts), 2 steps, resume checkpoint at step 2
#   3. resume from that checkpoint and train to step 4
#   4. eval generations from the step-4 HF export (FSDP full-state gather)
# Needs the data from prepare_data.slurm. On Newton, `normal` nodes have 2x H100 PCIe:
#   sbatch -p normal -A abedi --gres=gpu:nvidia_h100_pcie:2 \
#     --output=results/_pipeline_logs/%x-%j.out scripts/slurm/opd/smoke_2gpu.slurm
#SBATCH --job-name=opd-smoke2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/opd/lib/common.sh"
export EXPERIMENT=smoke2
export RUN_ID="${RUN_ID:-llama-3.2-1b-instruct__llama-3.1-8b-smoke-judge__healthbench__seed0__2gpu-$(date +%Y%m%d-%H%M)}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
NGPU="${NGPU:-2}"
RD="$(run_dir)"
mkdir -p "$RD/logs"
CONFIGS="configs/opd/base.yaml configs/opd/healthbench.yaml configs/opd/smoke.yaml"
COMMON_SET="experiment=$EXPERIMENT student.attn_implementation=sdpa teacher.attn_implementation=sdpa"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "== 1/4 FSDP2 loading check ($NGPU ranks)"
torchrun --standalone --nproc_per_node "$NGPU" -m rrh.opd.check_fsdp \
  --model "$(python -c "import yaml; print(yaml.safe_load(open('configs/opd/smoke.yaml'))['student']['path'])")"

echo "== 2/4 train steps 1-2 ($NGPU ranks, vLLM rollouts)"
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node "$NGPU" -m rrh.opd.train_cli --config $CONFIGS \
  --run_id "$RUN_ID" --results_root "$RESULTS_ROOT" --set $COMMON_SET train.total_steps=2 \
  2>&1 | tee "$RD/logs/train_part1.log"

echo "== 3/4 resume, train steps 3-4"
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node "$NGPU" -m rrh.opd.train_cli --config $CONFIGS \
  --run_id "$RUN_ID" --results_root "$RESULTS_ROOT" --set $COMMON_SET train.total_steps=4 \
  2>&1 | tee "$RD/logs/train_part2.log"

echo "== 4/4 eval generations from the step-4 HF export"
python -m rrh.opd.evaluate --run_dir "$RD" --steps 4

python - "$RD" <<'EOF'
import json, sys
from pathlib import Path
from rrh.results.layout import RunDir
rd = RunDir(sys.argv[1])
rows = [json.loads(l) for l in open(rd.train_metrics()) if l.strip()]
steps = [r["step"] for r in rows]
assert steps == [1, 2, 3, 4], steps
assert "resumed from" in (rd.path / "logs" / "train_part2.log").read_text(), "step 3 did not resume"
gens = sorted(rd.path.glob("generations/**/*step_000004*.jsonl"))
assert gens, "no eval generations for step 4"
n = sum(1 for g in gens for l in open(g) if l.strip())
assert n > 0, "empty eval generations"
loss = {r["step"]: r.get("loss") for r in rows}
fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
print("SMOKE2 OK", "steps", steps, "loss", {s: fmt(v) for s, v in loss.items()},
      "eval_generations", n)
EOF

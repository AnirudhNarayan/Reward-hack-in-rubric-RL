"""Generate policy responses for an eval set at one checkpoint.

python -m rrh.eval.generate_cli --model_path CKPT --data data/medical/eval300.jsonl \
    --step 25 --context prompt --n 1 --out runs/med_weak/gen/step25.jsonl
"""

from __future__ import annotations

import argparse

from rrh.eval.policy import PolicyRunner
from rrh.utils.io import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--context", choices=["prompt", "rubric"], default="prompt")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = read_jsonl(a.data)[: a.limit]
    runner = PolicyRunner(a.model_path, tensor_parallel_size=a.tp, seed=a.seed)
    outs = runner.generate(rows, a.context, n=a.n, temperature=a.temperature,
                           max_tokens=a.max_tokens, seed=a.seed)
    write_jsonl(a.out, ({"id": r["id"], "step": a.step, "context": a.context, "sample_idx": j,
                         "response": o["response"], "n_tokens": len(o["token_ids"]),
                         "finish_reason": o["finish_reason"]}
                        for r, samples in zip(rows, outs) for j, o in enumerate(samples)))


if __name__ == "__main__":
    main()

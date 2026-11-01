"""HealthBench (App. F): the fixed 1,000-example subset and canonical conversion.

Mirrors simple-evals: rng = random.Random(0); examples = rng.sample(examples, 1000).

python -m rrh.data.healthbench --out $DATA/healthbench/hb1000.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request

from rrh.utils.io import write_jsonl

URL = ("https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
       "2025-05-07-06-14-12_oss_eval.jsonl")


def load(n: int = 1000, seed: int = 0) -> list[dict]:
    with urllib.request.urlopen(URL) as f:
        examples = [json.loads(l) for l in f.read().decode().splitlines() if l.strip()]
    if n < len(examples):
        examples = random.Random(seed).sample(examples, n)
    return [{"id": f"hb-{e['prompt_id']}", "domain": "medical", "source": "healthbench",
             "prompt": e["prompt"], "criteria": [r["criterion"] for r in e["rubrics"]],
             "weights": [float(r["points"]) for r in e["rubrics"]],
             "tags": [r.get("tags", []) for r in e["rubrics"]]}
            for e in examples]


def healthbench_score(weights, met) -> float:
    """Original HealthBench per-example score: sum(points of met items) / sum(positive points).

    Not clipped per example; the run-level mean is clipped to [0, 1] (simple-evals).
    The 'flipped' variant (App. I.5) is rrh.grading.reward.rubric_reward.
    """
    pos = sum(w for w in weights if w > 0)
    return sum(w for w, m in zip(weights, met) if m) / pos if pos else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    rows = load(a.n)
    write_jsonl(a.out, rows)
    print(f"wrote {len(rows)} HealthBench examples")


if __name__ == "__main__":
    main()

"""Grade HealthBench generations with the canonical simple-evals per-item grader (App. F).

Uses GRADER_TEMPLATE from openai/simple-evals (cloned into third_party/simple-evals by
scripts/setup_env.sh) so the grading text is exactly the benchmark's; one call per rubric item,
graded by gpt-4.1-2025-04-14. Writes per-example original and flipped (App. I.5) scores.

python -m rrh.eval.healthbench_grade --generations runs/X/healthbench_gen/step50.jsonl \
    --data $DATA_DIR/healthbench/hb1000.jsonl --out runs/X/healthbench_scores.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np

from rrh.data.healthbench import healthbench_score
from rrh.grading.parsing import extract_json
from rrh.grading.reward import rubric_reward
from rrh.utils.io import read_jsonl, write_jsonl
from rrh.utils.runtime import REPO_ROOT, Runtime


def grader_template() -> str:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "simple-evals"))
    sys.path.insert(0, str(REPO_ROOT / "third_party"))
    try:
        from simple_evals.healthbench_eval import GRADER_TEMPLATE  # type: ignore
    except ImportError:
        from healthbench_eval import GRADER_TEMPLATE  # type: ignore
    return GRADER_TEMPLATE


def _conversation(prompt: list[dict], response: str) -> str:
    return "\n\n".join(f"{m['role']}: {m['content']}"
                       for m in prompt + [{"role": "assistant", "content": response}])


async def grade_item(client, template: str, convo: str, criterion: str, points: float,
                     max_retries: int = 5) -> tuple[bool, str]:
    rubric_item = f"[{points:g}] {criterion}"
    user = template.replace("<<conversation>>", convo).replace("<<rubric_item>>", rubric_item)
    for attempt in range(max_retries):
        try:
            obj = extract_json(await client.chat(None, user, sample_idx=attempt))
            if obj.get("criteria_met") in (True, False):
                return obj["criteria_met"], str(obj.get("explanation", ""))
        except ValueError:
            pass
    return False, "UNPARSED"


async def run(a):
    rt = Runtime()
    client = rt.client(a.grader)
    tmpl = grader_template()
    data = {r["id"]: r for r in read_jsonl(a.data)}
    gens = read_jsonl(a.generations)

    async def one(g):
        d = data[g["id"]]
        convo = _conversation(d["prompt"], g["response"])
        res = await asyncio.gather(*[grade_item(client, tmpl, convo, c, w)
                                     for c, w in zip(d["criteria"], d["weights"])])
        met = [m for m, _ in res]
        return {"id": g["id"], "step": g["step"], "met": met,
                "explanations": [e for _, e in res],
                "score_original": healthbench_score(d["weights"], met),
                "score_flipped": rubric_reward(d["weights"], met),
                "response_chars": len(g["response"])}

    rows = await asyncio.gather(*[one(g) for g in gens])
    write_jsonl(a.out, rows, append=True)
    step = gens[0]["step"] if gens else None
    print(json.dumps({"step": step,
                      "healthbench": float(np.clip(np.mean([r["score_original"] for r in rows]), 0, 1)),
                      "flipped": float(np.mean([r["score_flipped"] for r in rows]))}))
    print(rt.cost.summary())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grader", default="gpt-4.1")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()

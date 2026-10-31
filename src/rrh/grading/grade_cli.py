"""Grade generations with one or more judges (training verifier and/or panel). Resumable.

python -m rrh.grading.grade_cli --generations runs/med_weak/gen/step25.jsonl \
    --data data/medical/eval300.jsonl --judges gpt-4o-mini,gpt-5.4,gemini-3-pro,claude-opus-4.6 \
    --out runs/med_weak/judgments.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from rrh.grading.grader import RubricGrader
from rrh.utils.io import iter_jsonl, read_jsonl, write_jsonl
from rrh.utils.runtime import Runtime


def _key(r: dict) -> tuple:
    return (r["id"], r["step"], r.get("context", "prompt"), r.get("sample_idx", 0), r["judge"])


async def run(a):
    rt = Runtime(cache_path=a.cache)
    data = {r["id"]: r for r in read_jsonl(a.data)}
    gens = read_jsonl(a.generations)
    done = {_key(r) for r in iter_jsonl(a.out)} if os.path.exists(a.out) else set()
    judges = [j.strip() for j in a.judges.split(",") if j.strip()]

    async def one(g: dict, judge: str, grader: RubricGrader):
        d = data[g["id"]]
        res = await grader.grade(d["prompt"], g["response"], d["criteria"], d["weights"])
        row = {"id": g["id"], "step": g["step"], "context": g.get("context", "prompt"),
               "sample_idx": g.get("sample_idx", 0), "judge": judge, **res.to_dict()}
        write_jsonl(a.out, [row], append=True)

    tasks = []
    for judge in judges:
        grader = RubricGrader(rt.client(judge), max_parse_retries=a.max_parse_retries)
        for g in gens:
            if g["id"] not in data:
                continue
            k = (g["id"], g["step"], g.get("context", "prompt"), g.get("sample_idx", 0), judge)
            if k not in done:
                tasks.append(one(g, judge, grader))
    logging.info("grading %d (generation, judge) pairs", len(tasks))
    for i in range(0, len(tasks), a.chunk):
        await asyncio.gather(*tasks[i:i + a.chunk])
        logging.info("done %d/%d | cost %s", min(i + a.chunk, len(tasks)), len(tasks),
                     rt.cost.usd())
    print(rt.cost.summary())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--max_parse_retries", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=512)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()

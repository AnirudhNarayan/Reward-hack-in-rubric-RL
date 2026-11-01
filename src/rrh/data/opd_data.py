"""Builds the three RP-OPD datasets (paper §4, App. A) as canonical JSONL.

    python -m rrh.data.opd_data --out_dir $DATA_DIR/opd [--only healthbench researchqa rubrichub_science]

Writes, per dataset, train.jsonl and eval_heldout.jsonl (500); HealthBench also gets
eval_overlap.jsonl (500 prompts drawn from train, the Fig. 3 setting). Every row is
{"id", "prompt": [chat messages], "criteria": [...], "weights": [...], ...}. A manifest.json
records the sizes vs the paper's. Prompt-length filtering happens in the trainer, which has the
tokenizer (D14).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rrh.utils.io import write_jsonl

PAPER = {"healthbench": (4_500, 500), "researchqa": (16_961, 500), "rubrichub_science": (28_918, 500)}
N_EVAL = 500


def _dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def build_healthbench(seed: int) -> dict[str, list[dict]]:
    """5,000 public examples -> 4,500 train / 500 held-out (RuscaRL-style random split, D19)."""
    from rrh.data import healthbench
    rows = healthbench.load(n=10 ** 9)                    # all examples, file order
    rows = [r for r in _dedup(rows) if r["criteria"]]
    rng = random.Random(seed)
    rng.shuffle(rows)
    heldout, train = rows[:N_EVAL], rows[N_EVAL:]
    overlap = random.Random(seed + 1).sample(train, min(N_EVAL, len(train)))
    return {"train": train, "eval_heldout": heldout, "eval_overlap": overlap}


def build_researchqa(seed: int, cache_dir: str | None) -> dict[str, list[dict]]:
    from rrh.data import researchqa
    s = researchqa.load_splits(cache_dir)
    return {"train": s["train"], "eval_heldout": random.Random(seed).sample(s["test"], N_EVAL)}


def build_rubrichub_science(seed: int, cache_dir: str | None) -> dict[str, list[dict]]:
    """Whole RuRL Science file (no source filter in this paper): 500 eval, rest (<= 28,918) train."""
    import pandas as pd

    from rrh.data import rubrichub
    df = pd.read_parquet(rubrichub._download(rubrichub.RL_FILES["science"], cache_dir))
    rows = [c for c in (rubrichub.to_canonical(r, "science", None) for r in df.to_dict("records")) if c]
    rows = _dedup(rows)
    for r in rows:
        r["source"] = "rubrichub_science"
    random.Random(seed).shuffle(rows)
    n_train = PAPER["rubrichub_science"][0]
    return {"eval_heldout": rows[:N_EVAL], "train": rows[N_EVAL:N_EVAL + n_train]}


def build(out_dir: str, only: list[str], seed: int, cache_dir: str | None) -> dict:
    out = Path(out_dir)
    manifest = {}
    builders = {"healthbench": lambda: build_healthbench(seed),
                "researchqa": lambda: build_researchqa(seed, cache_dir),
                "rubrichub_science": lambda: build_rubrichub_science(seed, cache_dir)}
    for name in only:
        splits = builders[name]()
        for split, rows in splits.items():
            write_jsonl(out / name / f"{split}.jsonl", rows)
        train_ids = {r["id"] for r in splits["train"]}
        leak = sum(r["id"] in train_ids for r in splits["eval_heldout"])
        manifest[name] = {"sizes": {k: len(v) for k, v in splits.items()},
                          "paper": {"train": PAPER[name][0], "eval": PAPER[name][1]},
                          "heldout_overlap_with_train": leak, "seed": seed}
        print(f"[{name}] " + " ".join(f"{k}={len(v)}" for k, v in splits.items()) +
              f" (paper train={PAPER[name][0]}, eval={PAPER[name][1]}); heldout∩train={leak}")
    prev = out / "manifest.json"
    full = json.loads(prev.read_text()) if prev.exists() else {}
    full.update(manifest)
    out.mkdir(parents=True, exist_ok=True)
    prev.write_text(json.dumps(full, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--only", nargs="+", default=list(PAPER))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=None)
    a = ap.parse_args()
    build(a.out_dir, a.only, a.seed, a.cache_dir)


if __name__ == "__main__":
    main()

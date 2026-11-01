"""RubricHub -> canonical JSONL + verl parquet, with paper-sized splits (Sec. 2.2).

Source: HF dataset ``sojuL/RubricHub_v1`` (RL files ``RuRL/rurbichub_v1_{Medical,Science}.parquet``).
Each RL row: prompt (chat list), data_source, ability,
reward_model {ground_truth: "", style: "rubric", rubrics: [{criterion, points, tags}]}.

The paper keeps prompts from II-medical-reasoning (medical) and RaR-science / ResearchQA /
MegaScience (science). The RL files carry no source field, so (optionally) we label rows
by joining prompt text against the SFT files' ``source`` column and keep matching rows; the
result is then subsampled to the paper's split sizes. See REPLICATION_NOTES.md.

python -m rrh.data.rubrichub --out_dir $DATA/rubrichub --source_filter
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from rrh.utils.io import write_jsonl

HF_REPO = "sojuL/RubricHub_v1"
RL_FILES = {"medical": "RuRL/rurbichub_v1_Medical.parquet",
            "science": "RuRL/rurbichub_v1_Science.parquet"}
SFT_FILES = ["sft_RuFT/rurbichub_v1_6samples_156k_sft_data.parquet",
             "sft_RuFT/rurbichub_v1_best_of_6samples_26k_sft_data.parquet"]
PAPER_SOURCES = {"medical": ("ii-medical",),
                 "science": ("rar_science", "researchqa", "megescience", "megascience")}
PAPER_SIZES = {"medical": (12_519, 1_391), "science": (19_806, 2_201)}


def _download(fname: str, cache_dir: str | None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, fname, repo_type="dataset", cache_dir=cache_dir)


def _pid(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _user_text(prompt) -> str:
    return "\n".join(m["content"] for m in prompt if m["role"] == "user")


def load_sources(cache_dir: str | None) -> dict[str, str]:
    import pandas as pd
    src: dict[str, str] = {}
    for f in SFT_FILES:
        df = pd.read_parquet(_download(f, cache_dir), columns=["query", "source"])
        for q, s in zip(df["query"], df["source"]):
            src.setdefault(q.strip(), str(s))
    return src


def to_canonical(row, domain: str, sources: dict[str, str] | None) -> dict | None:
    prompt = [dict(role=m["role"], content=m["content"]) for m in row["prompt"]]
    rubrics = list(row["reward_model"]["rubrics"])
    criteria = [r["criterion"] for r in rubrics]
    weights = [float(r["points"]) for r in rubrics]
    if not criteria or sum(abs(w) for w in weights) == 0:
        return None
    text = _user_text(prompt)
    return {"id": f"{domain}-{_pid(text)}", "domain": domain,
            "source": (sources or {}).get(text.strip(), "unknown"),
            "prompt": prompt, "criteria": criteria, "weights": weights}


def to_verl(row: dict, split: str) -> dict:
    """verl RL parquet row. Rubric travels in extra_info for the custom reward function."""
    return {"data_source": f"rubrichub_{row['domain']}", "prompt": row["prompt"],
            "ability": row["domain"],
            "reward_model": {"style": "rubric", "ground_truth": ""},
            # compute_score does not receive the prompt, so the verifier's copy rides here.
            "extra_info": {"id": row["id"], "split": split, "criteria": row["criteria"],
                           "weights": row["weights"], "prompt_messages": row["prompt"]}}


def build(out_dir: str, source_filter: bool, seed: int, cache_dir: str | None,
          n_eval: int, n_verifier_sel: int) -> None:
    import pandas as pd
    out = Path(out_dir)
    sources = load_sources(cache_dir) if source_filter else None
    for domain, fname in RL_FILES.items():
        df = pd.read_parquet(_download(fname, cache_dir))
        rows, seen = [], set()
        for r in df.to_dict("records"):
            c = to_canonical(r, domain, sources)
            if c is None or c["id"] in seen:
                continue
            seen.add(c["id"])
            rows.append(c)
        n_all = len(rows)
        if source_filter:
            keep = PAPER_SOURCES[domain]
            rows = [r for r in rows if any(k in r["source"].lower() for k in keep)]
        n_train, n_test = PAPER_SIZES[domain]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if len(rows) < n_train + n_test:  # keep the paper's ~90/10 ratio on what we have
            n_test = round(len(rows) * n_test / (n_train + n_test))
            n_train = len(rows) - n_test
        train, test = rows[:n_train], rows[n_train:n_train + n_test]
        d = out / domain
        write_jsonl(d / "train.jsonl", train)
        write_jsonl(d / "test.jsonl", test)
        eval_rows = random.Random(seed).sample(test, min(n_eval, len(test)))
        write_jsonl(d / "eval300.jsonl", eval_rows)
        # Sec. 4 analyses use 500 prompts (Tables 10-16); drawn from test, superset-independent.
        write_jsonl(d / "sec4_500.jsonl", random.Random(seed + 2).sample(test, min(500, len(test))))
        write_jsonl(d / "verifier_selection.jsonl",
                    random.Random(seed + 1).sample(train, min(n_verifier_sel, len(train))))
        pd.DataFrame([to_verl(r, "train") for r in train]).to_parquet(d / "train.parquet")
        pd.DataFrame([to_verl(r, "test") for r in test]).to_parquet(d / "test.parquet")
        pd.DataFrame([to_verl(r, "test") for r in eval_rows]).to_parquet(d / "eval300.parquet")
        print(f"[{domain}] raw={n_all} kept={len(rows)} train={len(train)} test={len(test)} "
              f"(paper {PAPER_SIZES[domain]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--source_filter", action="store_true",
                    help="keep only prompts traced to the paper's source datasets via the SFT join")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--n_verifier_sel", type=int, default=1000)
    a = ap.parse_args()
    build(a.out_dir, a.source_filter, a.seed, a.cache_dir, a.n_eval, a.n_verifier_sel)


if __name__ == "__main__":
    main()

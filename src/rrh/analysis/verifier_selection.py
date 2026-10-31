"""Verifier-selection table (Sec. 2.3, Tables 1 & 4) from base-policy judgments.

Steps: generate base-policy responses on data/<domain>/verifier_selection.jsonl (1,000 train
prompts), grade them with every candidate + the panel (grade_cli), then:

python -m rrh.analysis.verifier_selection --judgments runs/versel_medical/judgments.jsonl \
    --candidates gpt-5,gpt-oss-120b,gpt-5-mini,gpt-oss-20b,gpt-5-nano,gpt-4o-mini,llama-3.1-8b-judge
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from rrh.analysis.verifier_agreement import candidate_vs_panel
from rrh.utils.io import read_jsonl

PANEL = ["gpt-5.4", "gemini-3-pro", "claude-opus-4.6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--panel", default=",".join(PANEL))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    panel = a.panel.split(",")
    idx = defaultdict(dict)
    for r in read_jsonl(a.judgments):
        idx[(r["id"], r["step"], r.get("sample_idx", 0))][r["judge"]] = r["met"]
    table = {"panel_members_vs_others": {m: candidate_vs_panel(idx, m, panel) for m in panel},
             "candidates_vs_panel": {c: candidate_vs_panel(idx, c, panel)
                                     for c in a.candidates.split(",")}}
    print(f"{'verifier':<22}{'agree%':>8}{'FP%':>7}{'FN%':>7}{'n':>9}")
    for block in table.values():
        for name, s in block.items():
            if s:
                print(f"{name:<22}{s['agreement']:>8.1f}{s['fp']:>7.1f}{s['fn']:>7.1f}{s['n']:>9}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()

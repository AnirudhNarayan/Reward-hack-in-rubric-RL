"""Validation of the judge substitutions (DECISIONS.md D10-D12).

Criterion-level agreement / FP / FN of each judge against the strong curve judge (default
Llama-3.3-70B, our Qwen3-32B stand-in) on the same (prompt, response) pairs, plus each judge's mean
Eq. (1) score on the common set -- the quantity the paper's Fig. 13 compares across judges.

    python -m rrh.analysis.judge_agreement --run_dirs results/rp_opd/<run> [...] \
        --reference llama-3.3-70b-judge \
        --judges gpt-oss-120b-ref gemma-3-27b-judge gemma-3-27b-judge-shortcut \
        --out_dir results/judge_validation

FP = judge says met, reference says not met (% of all criteria); FN the converse
(rrh.analysis.verifier_agreement.agreement_stats). For "-shortcut" judgments, ``standard_met`` is
compared (the ordinary-grader decision); the shortcut-aware score is reported alongside.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rrh.analysis.verifier_agreement import agreement_stats
from rrh.grading.reward import normalized_rubric_score
from rrh.results.layout import RunDir
from rrh.utils.io import read_jsonl

# Paper Fig. 13 (same 500 responses per checkpoint; Qwen3-32B vs GPT-4o-mini). Values as printed
# in the figure, grouped per checkpoint as (Qwen3-32B, 4o-mini standard, 4o-mini shortcut-aware);
# the grouping is our reading of the bar labels.
PAPER_FIG13 = {"RubricHub Science OPD+RL": (0.820, 0.847, 0.829),
               "RubricHub Science SFT+RL": (0.808, 0.832, 0.815),
               "ResearchQA OPD@40+RL": (0.850, 0.817, 0.797),
               "ResearchQA SFT@80+RL": (0.774, 0.744, 0.721)}


def load_judgments(run: RunDir, judge_dir: str) -> dict[tuple, dict]:
    out = {}
    base = run.path / "judgments" / judge_dir
    if not base.exists():
        return out
    for f in base.glob("*/*.jsonl"):
        evalset, step = f.parent.name, f.stem
        for j in read_jsonl(f):
            out[(run.run_id, evalset, step, j["id"], j.get("sample_idx", 0))] = j
    return out


def _met(j: dict) -> list:
    return j.get("standard_met", j["met"])


def _weights(run: RunDir) -> dict[str, list[float]]:
    cfg = run.read_config()
    w = {}
    for path in cfg["eval"]["eval_sets"].values():
        for r in read_jsonl(path):
            w[r["id"]] = r["weights"]
    return w


def compare(runs: list[RunDir], reference: str, judges: list[str]) -> list[dict]:
    ref, weights = {}, {}
    for r in runs:
        ref.update(load_judgments(r, reference))
        weights.update(_weights(r))
    rows = []
    for jd in judges:
        other = {}
        for r in runs:
            other.update(load_judgments(r, jd))
        keys = sorted(k for k in other if k in ref)
        if not keys:
            rows.append({"judge": jd, "n_responses": 0})
            continue
        preds, refs, pol = [], [], []
        s_ref, s_std, s_final = [], [], []
        for k in keys:
            a, b = _met(other[k]), ref[k]["met"]
            if len(a) != len(b):
                continue
            w = weights[k[3]]
            preds.append([1 if x else 0 for x in a])
            refs.append([1 if x else 0 for x in b])
            pol.append([1 if x > 0 else 0 for x in w])
            s_ref.append(normalized_rubric_score(w, b))
            s_std.append(normalized_rubric_score(w, a))
            s_final.append(other[k].get("score", s_std[-1]))
        P, R, Pol = (np.concatenate(x) for x in (preds, refs, pol))
        row = {"judge": jd, "reference": reference, "n_responses": len(s_ref), "n_criteria": int(len(P))}
        row.update({f"all_{k}": v for k, v in agreement_stats(P, R).items() if k != "n"})
        for name, m in (("pos", Pol == 1), ("neg", Pol == 0)):
            if m.any():
                row.update({f"{name}_{k}": v for k, v in agreement_stats(P[m], R[m]).items()})
        row.update({"score_reference": float(np.mean(s_ref)), "score_judge_standard": float(np.mean(s_std)),
                    "score_judge_final": float(np.mean(s_final)),
                    "score_gap_standard_minus_ref": float(np.mean(s_std) - np.mean(s_ref))})
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "metrics" / "judge_agreement.json").write_text(json.dumps(
        {"rows": rows, "paper_fig13_reference": PAPER_FIG13}, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "judge_agreement.csv", index=False)
    cols = [c for c in ["judge", "n_responses", "n_criteria", "all_agreement", "all_fp", "all_fn",
                        "score_reference", "score_judge_standard", "score_judge_final",
                        "score_gap_standard_minus_ref"] if c in df]
    (out_dir / "tables" / "judge_agreement.tex").write_text(
        df[cols].to_latex(index=False, float_format="%.3f"))
    paper = pd.DataFrame([{"checkpoint": k, "qwen3_32b": v[0], "gpt4o_mini_standard": v[1],
                           "gpt4o_mini_shortcut_aware": v[2], "gap_standard_minus_qwen": v[1] - v[0]}
                          for k, v in PAPER_FIG13.items()])
    paper.to_csv(out_dir / "tables" / "paper_fig13_reference.csv", index=False)
    print(df[cols].to_string(index=False))
    print("\nPaper Fig. 13 (reference):\n" + paper.to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--reference", default="llama-3.3-70b-judge")
    ap.add_argument("--judges", nargs="+", default=["gpt-oss-120b-ref", "gemma-3-27b-judge",
                                                    "gemma-3-27b-judge-shortcut"])
    ap.add_argument("--out_dir", default="results/judge_validation")
    a = ap.parse_args(argv)
    rows = compare([RunDir(p) for p in a.run_dirs], a.reference, a.judges)
    write_outputs(rows, Path(a.out_dir))


if __name__ == "__main__":
    main()

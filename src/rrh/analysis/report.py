"""Per-run Section 3 report: reward curves (Fig. 1 cols 1-2), exploitation rate (Fig. 1 col 3),
self-internalization gap (Fig. 4, App. C), App. G.1 / G.3 checks, HealthBench (Fig. 2, Table 7),
and optionally the failure-mode taxonomy (Fig. 3, App. H; API calls).

python -m rrh.analysis.report --run_dir $RUNS_DIR/medical_gpt-4o-mini_... \
    --data $DATA_DIR/medical/eval300.jsonl --train_judge gpt-4o-mini [--taxonomy]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from rrh.analysis import exploitation as ex
from rrh.analysis.self_gap import compare_with_rewards, self_gap_by_step
from rrh.grading.reward import rubric_reward, unanimous_consensus
from rrh.utils.io import read_jsonl, write_jsonl

PANEL = ["gpt-5.4", "gemini-3-pro", "claude-opus-4.6"]


def _prompt_only(rows):
    return [r for r in rows if r.get("context", "prompt") == "prompt" and r.get("sample_idx", 0) == 0]


def length_growth(run_dir: Path) -> dict:
    by_step = defaultdict(list)
    for f in glob.glob(str(run_dir / "gen" / "step*.jsonl")):
        for r in read_jsonl(f):
            by_step[r["step"]].append(len(r["response"]))
    steps = sorted(by_step)
    if not steps:
        return {}
    mean = {s: float(np.mean(by_step[s])) for s in steps}
    return {"mean_chars": mean, "growth_x": mean[steps[-1]] / mean[steps[0]]}


def rubric_conditioned_check(run_dir: Path, data: dict, panel, rule) -> list[dict]:
    """App. G.1 / Table 8: consensus reward of rubric-conditioned samples per step."""
    p = run_dir / "judgments_rubric_cond.jsonl"
    if not p.exists():
        return []
    idx = ex.index_judgments(read_jsonl(p))
    out = []
    for t in sorted(idx):
        vals = [rubric_reward(data[pid]["weights"],
                              unanimous_consensus([j[m]["met"] for m in panel], rule))
                for pid, j in idx[t].items() if all(m in j for m in panel)]
        out.append({"step": t, "rc_reward": float(np.mean(vals)) if vals else float("nan"),
                    "n": len(vals)})
    return out


def healthbench_curve(run_dir: Path) -> list[dict]:
    p = run_dir / "healthbench_scores.jsonl"
    if not p.exists():
        return []
    by = defaultdict(list)
    for r in read_jsonl(p):
        by[r["step"]].append(r)
    return [{"step": s, "healthbench": float(np.clip(np.mean([r["score_original"] for r in v]), 0, 1)),
             "flipped": float(np.mean([r["score_flipped"] for r in v])),
             "mean_chars": float(np.mean([r["response_chars"] for r in v])), "n": len(v)}
            for s, v in sorted(by.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train_judge", required=True)
    ap.add_argument("--panel", default=",".join(PANEL))
    ap.add_argument("--consensus_rule", default="all_met", choices=["all_met", "not_all_reject"])
    ap.add_argument("--taxonomy", action="store_true", help="run App. H extraction+classification")
    ap.add_argument("--extractor", default="gpt-5.4")
    ap.add_argument("--classifier", default="gpt-5.4-nano")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    panel = a.panel.split(",")
    data = {r["id"]: r for r in read_jsonl(a.data)}
    judg = _prompt_only(read_jsonl(run_dir / "judgments.jsonl"))
    idx = ex.index_judgments(judg)
    steps = sorted(idx)

    rep: dict = {"run": run_dir.name, "train_judge": a.train_judge, "panel": panel,
                 "consensus_rule": a.consensus_rule, "steps": steps}
    rep["rewards"] = ex.reward_curves(idx, data, a.train_judge, panel, steps, a.consensus_rule,
                                      n_boot=a.n_boot)
    terms = ex.per_prompt_terms(idx, data, a.train_judge, panel, steps)
    rep["exploitation"] = ex.exploitation_rate(terms, n_boot=a.n_boot)
    instances = ex.exploited_instances(idx, data, a.train_judge, panel, steps)
    rep["n_exploited_instances"] = len(instances)
    write_jsonl(run_dir / "exploited_instances.jsonl", instances)

    sg_rows = [r for f in glob.glob(str(run_dir / "selfgap" / "step*.jsonl")) for r in read_jsonl(f)]
    if sg_rows:
        rep["self_gap"] = self_gap_by_step(sg_rows, n_boot=a.n_boot)
        rep["self_gap_vs_rewards"] = compare_with_rewards(rep["self_gap"], rep["rewards"])
    rep["rubric_conditioned"] = rubric_conditioned_check(run_dir, data, panel, a.consensus_rule)
    rep["length"] = length_growth(run_dir)
    rep["healthbench"] = healthbench_curve(run_dir)

    if a.taxonomy and instances:
        from rrh.analysis.taxonomy import run_taxonomy, summarize
        from rrh.utils.runtime import Runtime
        rt = Runtime()
        labeled = asyncio.run(run_taxonomy(rt.client(a.extractor), rt.client(a.classifier),
                                           instances))
        write_jsonl(run_dir / "taxonomy.jsonl", labeled)
        rep["taxonomy"] = summarize(labeled)
        print(rt.cost.summary())

    with open(run_dir / "report.json", "w") as f:
        json.dump(rep, f, indent=2)
    from rrh.analysis.plots import plot_run
    plot_run(rep, run_dir / "figures")
    print(json.dumps({k: rep[k] for k in ("run", "n_exploited_instances")}, indent=2))
    print(f"wrote {os.path.join(run_dir, 'report.json')} and figures/")


if __name__ == "__main__":
    main()

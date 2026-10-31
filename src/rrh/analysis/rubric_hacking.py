"""Section 4 report ("hacking the rubric, not the verifier") for one run.

Inputs in --run_dir (produced with --data = the 500-prompt Sec. 4 set):
  gen_sec4/step*.jsonl          prompt-only generations at the analysed checkpoints (paper: 8)
  judgments_sec4.jsonl          panel rubric grades of those generations (grade_cli)
Outputs: sec4_report.json with
  - Table 10/18: rubric-based vs rubric-free panel winners (majority / consensus)
  - Table 11/12/19: rubric-free dimension scores base vs ckpt-last (overall and per judge)
  - Table 13/20: rubric weight by category;  Table 2/14/21: satisfaction by category
  - Table 15/16/22: within-prompt fixed-effects correlations with length and claim counts

python -m rrh.analysis.rubric_hacking --run_dir $RUNS_DIR/<run> --data $DATA_DIR/medical/sec4_500.jsonl \
    --domain medical [--claims]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rrh.analysis import pairwise as pw
from rrh.analysis import rubric_types as rt_
from rrh.analysis.claims import extract_all, fixed_effects_corr
from rrh.grading.reward import rubric_reward
from rrh.utils.io import read_jsonl, write_jsonl
from rrh.utils.runtime import Runtime

PANEL = ["gpt-5.4", "gemini-3-pro", "claude-opus-4.6"]
SAT_KEYS = rt_.CATS + ["presence_total", "absence_total", "other_total", "total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--domain", required=True, choices=["medical", "science"])
    ap.add_argument("--panel", default=",".join(PANEL))
    ap.add_argument("--category_classifier", default="gpt-5.4")
    ap.add_argument("--claims", action="store_true", help="run claim extraction (API)")
    ap.add_argument("--claim_model", default="gpt-5.4")
    a = ap.parse_args()

    run_dir, panel = Path(a.run_dir), a.panel.split(",")
    data = {r["id"]: r for r in read_jsonl(a.data)}
    gens = defaultdict(dict)  # step -> id -> response
    for f in glob.glob(str(run_dir / "gen_sec4" / "step*.jsonl")):
        for g in read_jsonl(f):
            gens[g["step"]][g["id"]] = g["response"]
    steps = sorted(gens)
    base_step, last_step = steps[0], steps[-1]
    grades = defaultdict(dict)  # (step, id) -> judge -> met
    for r in read_jsonl(run_dir / "judgments_sec4.jsonl"):
        grades[(r["step"], r["id"])][r["judge"]] = r["met"]

    rt = Runtime()
    rep: dict = {"run": run_dir.name, "steps": steps, "base_step": base_step,
                 "last_step": last_step}

    # ---- rubric categories (Table 13) ----
    cat_path = run_dir.parent / f"rubric_categories_{Path(a.data).stem}.json"
    if cat_path.exists():
        labels = json.loads(cat_path.read_text())
    else:
        labels = asyncio.run(rt_.classify_all(rt.client(a.category_classifier), list(data.values())))
        cat_path.write_text(json.dumps(labels))
    rep["category_weight_share"] = rt_.weight_shares(list(data.values()), labels)
    rep["n_rubric_items"] = sum(len(r["criteria"]) for r in data.values())

    # ---- per-(step, prompt) satisfaction by category, fractional over judges ----
    sat_rows = []
    for (step, pid), jm in grades.items():
        if pid not in data or not all(m in jm for m in panel):
            continue
        frac = rt_.fractional_satisfaction(data[pid], [jm[m] for m in panel])
        sat_rows.append({"step": step, "id": pid, "chars": len(gens[step].get(pid, "")),
                         **rt_.per_prompt_satisfaction(data[pid], labels[pid], frac)})
    rep["satisfaction_by_step"] = {
        s: {k: float(np.nanmean([r[k] for r in sat_rows if r["step"] == s])) for k in SAT_KEYS}
        for s in steps}
    rep["mean_chars_by_step"] = {s: float(np.mean([len(v) for v in gens[s].values()])) for s in steps}

    # ---- rubric-based vs rubric-free pairwise (base vs ckpt-last) ----
    pairs = [{"id": pid, "question": data[pid]["prompt"], "base": gens[base_step][pid],
              "ckpt": gens[last_step][pid]}
             for pid in data if pid in gens[base_step] and pid in gens[last_step]]
    free = {}
    for judge in panel:
        free[judge] = asyncio.run(pw.judge_all(rt.client(judge), a.domain, pairs))
        write_jsonl(run_dir / f"pairwise_free_{judge}.jsonl", free[judge])
    all_judged = [r for v in free.values() for r in v]
    rep["rubric_free_dimensions"] = pw.dimension_summary(all_judged)
    rep["rubric_free_dimensions_per_judge"] = {j: pw.dimension_summary(v) for j, v in free.items() if v}
    free_votes = defaultdict(list)
    for j in panel:
        for r in free[j]:
            free_votes[r["id"]].append(pw.winner(r["base"]["overall"], r["ckpt"]["overall"]))
    rubric_votes = defaultdict(list)
    for p in pairs:
        gb, gc = grades.get((base_step, p["id"]), {}), grades.get((last_step, p["id"]), {})
        if all(m in gb and m in gc for m in panel):
            w = data[p["id"]]["weights"]
            rubric_votes[p["id"]] = [pw.winner(rubric_reward(w, gb[m]), rubric_reward(w, gc[m]))
                                     for m in panel]
    full = {k: v for k, v in free_votes.items() if len(v) == len(panel)}
    rep["rubric_vs_free"] = {rule: pw.agreement_table(rubric_votes, full, rule)
                             for rule in ("majority", "consensus")}
    for rule in ("majority", "consensus"):
        rw = [pw.aggregate_winner(v, rule) for v in rubric_votes.values()]
        fw = [pw.aggregate_winner(v, rule) for v in full.values()]
        rep[f"rubric_prefers_ckpt_{rule}"] = float(np.mean([x == "ckpt" for x in rw if x]))
        rep[f"free_prefers_base_{rule}"] = float(np.mean([x == "base" for x in fw if x]))

    # ---- claims + within-prompt fixed effects (Tables 15/16) ----
    ids = np.asarray([r["id"] for r in sat_rows])
    rep["fe_length"] = {k: fixed_effects_corr(ids, [r[k] for r in sat_rows],
                                              [r["chars"] for r in sat_rows]) for k in SAT_KEYS}
    if a.claims:
        items = [{"id": pid, "step": s, "question": data[pid]["prompt"], "response": resp}
                 for s in steps for pid, resp in gens[s].items() if pid in data]
        claims = asyncio.run(extract_all(rt.client(a.claim_model), a.domain, items))
        write_jsonl(run_dir / "claims.jsonl", claims)
        cl = {(c["step"], c["id"]): c for c in claims}
        rows = [(r, cl[(r["step"], r["id"])]) for r in sat_rows if (r["step"], r["id"]) in cl]
        cids = np.asarray([r["id"] for r, _ in rows])
        rep["fe_claims"] = {
            k: {y: fixed_effects_corr(cids, [r[k] for r, _ in rows], [c[y] for _, c in rows])
                for y in ("total_claims", "incorrect_claims", "error_rate")}
            for k in SAT_KEYS}
        rep["claims_by_step"] = {s: {y: float(np.nanmean([c[y] for c in claims if c["step"] == s]))
                                     for y in ("total_claims", "incorrect_claims", "error_rate")}
                                 for s in steps}

    (run_dir / "sec4_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(rt.cost.summary())


if __name__ == "__main__":
    main()

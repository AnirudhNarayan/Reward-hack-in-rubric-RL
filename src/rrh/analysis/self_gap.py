"""Self-internalization gap (Sec. 3.3) statistics.

Raw input rows (from rrh.eval.self_gap_score): one per (id, step, sample_idx) with
per-token average log-probs ``l_cond`` and ``l_prompt`` of the same rubric-conditioned
sample under the two contexts.
    Delta(t) = mean_{i,j} (l_prompt - l_cond)        (<= 0 in expectation)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats


def self_gap_by_step(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_step[r["step"]][r["id"]].append(r["l_prompt"] - r["l_cond"])
    out = []
    for t, per_prompt in sorted(by_step.items()):
        # Equal K per prompt makes the grand mean equal the mean of per-prompt means;
        # resample prompts (the unit of the 300-prompt eval set) for the CI.
        sums = np.asarray([sum(v) for v in per_prompt.values()])
        cnts = np.asarray([len(v) for v in per_prompt.values()])
        delta = sums.sum() / cnts.sum()
        bs = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(sums), len(sums))
            bs.append(sums[ix].sum() / cnts[ix].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append({"step": t, "self_gap": float(delta), "ci_lo": float(lo), "ci_hi": float(hi),
                    "n_prompts": len(sums), "n_samples": int(cnts.sum())})
    return out


def compare_with_rewards(gap: list[dict], rewards: list[dict]) -> dict:
    """Pearson r of Delta(t) vs consensus / training-verifier reward and argmax steps."""
    g = {r["step"]: r["self_gap"] for r in gap}
    ref = {r["step"]: r["ref_reward"] for r in rewards}
    prx = {r["step"]: r["proxy_reward"] for r in rewards}
    steps = sorted(set(g) & set(ref) & set(prx))
    gv = np.asarray([g[s] for s in steps])
    rv = np.asarray([ref[s] for s in steps])
    pv = np.asarray([prx[s] for s in steps])
    res = {"steps": steps,
           "argmax_self_gap": steps[int(np.argmax(gv))],
           "argmax_consensus": steps[int(np.argmax(rv))],
           "argmax_train_verifier": steps[int(np.argmax(pv))]}
    if len(steps) >= 3:
        res["pearson_gap_vs_consensus"] = float(stats.pearsonr(gv, rv)[0])
        res["pearson_gap_vs_train"] = float(stats.pearsonr(gv, pv)[0])
    peak = rv.max()
    # Stopping regret (App. C): consensus reward forgone by stopping at each metric's argmax.
    res["regret_self_gap"] = float(peak - ref[res["argmax_self_gap"]])
    res["regret_train_verifier"] = float(peak - ref[res["argmax_train_verifier"]])
    return res

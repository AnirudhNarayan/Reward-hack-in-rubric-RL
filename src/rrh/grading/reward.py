"""Rubric reward aggregation (Sec. 2.1) and panel consensus (Sec. 2.2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rubric_reward(weights: Sequence[float], met: Sequence[bool | int | None]) -> float:
    """R = [sum_{w>0} w*g + sum_{w<0} |w|*(1-g)] / sum |w|, in [0, 1].

    ``None`` judgments (grader failed after retries) count as not met (g = 0), the
    conservative choice for positive items. For negative items this credits avoidance;
    unparseable outputs are rare and are logged by the grader.
    """
    w = np.asarray(weights, dtype=float)
    g = np.asarray([1.0 if m else 0.0 for m in met], dtype=float)
    denom = np.abs(w).sum()
    if denom == 0:
        return 0.0
    num = (w[w > 0] * g[w > 0]).sum() + (np.abs(w[w < 0]) * (1.0 - g[w < 0])).sum()
    return float(num / denom)


def normalized_rubric_score(weights: Sequence[float], met: Sequence[bool | int | None]) -> float:
    """"OPD Before RL" Eq. (1): R = sum_j w_j g_j / sum_{w_j>0} w_j.

    Met pitfalls (w < 0, g = 1) subtract, so R can be negative (HealthBench). This is the
    HealthBench score and differs from ``rubric_reward`` (sum |w| denominator). ``None`` -> g = 0.
    """
    w = np.asarray(weights, dtype=float)
    g = np.asarray([1.0 if m else 0.0 for m in met], dtype=float)
    pos = w[w > 0].sum()
    return float((w * g).sum() / pos) if pos > 0 else 0.0


def item_satisfied(weight: float, met: bool | None) -> bool:
    """Per-item 'satisfied' in the flipped sense (App. I.5): positive met, negative not met."""
    return bool(met) if weight > 0 else not bool(met)


def unanimous_consensus(panel_met: Sequence[Sequence[bool | None]],
                        rule: str = "all_met") -> list[bool]:
    """Reference judgment per criterion for R_ref (Sec. 2.2: "unanimous consensus").

    The paper does not pin down how a split panel is scored inside R_ref, so the rule
    is configurable (see REPLICATION_NOTES.md):
      - "all_met"          (default): met iff all judges say met.
      - "not_all_reject":  met unless all judges reject, i.e. 1 - J (App. E reading).

    panel_met: shape (n_judges, n_items).
    """
    arr = np.asarray([[1 if m else 0 for m in row] for row in panel_met])
    if rule == "all_met":
        return [bool(x) for x in (arr.sum(axis=0) == arr.shape[0])]
    if rule == "not_all_reject":
        return [bool(x) for x in (arr.sum(axis=0) > 0)]
    raise ValueError(f"unknown consensus rule {rule!r}")


def unanimous_reject(panel_met: Sequence[Sequence[bool | None]]) -> list[bool]:
    """J_ik = 1[sum_m g^m_ik = 0]."""
    arr = np.asarray([[1 if m else 0 for m in row] for row in panel_met])
    return [bool(x) for x in (arr.sum(axis=0) == 0)]


def majority_vote(panel_met: Sequence[Sequence[bool | None]]) -> list[bool]:
    """Panel majority (used for verifier selection, Table 1/4)."""
    arr = np.asarray([[1 if m else 0 for m in row] for row in panel_met])
    return [bool(x) for x in (arr.sum(axis=0) * 2 > arr.shape[0])]

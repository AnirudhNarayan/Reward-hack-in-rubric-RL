"""Verifier selection (Sec. 2.3, Tables 1 & 4) and panel-vs-human macro-F1 (App. E).

Agreement / FP / FN are criterion-level, against the panel majority vote. For panel
members, the reference is the majority of the OTHER two judges (a tie between two
judges counts as "not met", the panel's conservative direction).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _arr(met: Sequence) -> np.ndarray:
    return np.asarray([1 if m else 0 for m in met], dtype=np.int8)


def agreement_stats(pred: np.ndarray, ref: np.ndarray) -> dict:
    """FP = pred met & ref not met, as % of all criteria (so agree + FP + FN = 100)."""
    n = len(pred)
    return {"agreement": 100 * float((pred == ref).mean()),
            "fp": 100 * float(((pred == 1) & (ref == 0)).sum() / n),
            "fn": 100 * float(((pred == 0) & (ref == 1)).sum() / n), "n": n}


def candidate_vs_panel(idx_by_item: dict, candidate: str, panel: Sequence[str]) -> dict:
    """idx_by_item[(id, ...)][judge] = met list. Reference = panel majority (or others' majority)."""
    preds, refs = [], []
    others = [m for m in panel if m != candidate]
    for judges in idx_by_item.values():
        if candidate not in judges or any(m not in judges for m in others):
            continue
        votes = np.stack([_arr(judges[m]) for m in others])
        ref = (votes.sum(0) * 2 > len(others)).astype(np.int8)
        preds.append(_arr(judges[candidate]))
        refs.append(ref)
    if not preds:
        return {}
    return agreement_stats(np.concatenate(preds), np.concatenate(refs))


def macro_f1(pred: Sequence[int], gold: Sequence[int]) -> float:
    pred, gold = np.asarray(pred), np.asarray(gold)
    f1s = []
    for c in (0, 1):
        tp = ((pred == c) & (gold == c)).sum()
        fp = ((pred == c) & (gold != c)).sum()
        fn = ((pred != c) & (gold == c)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return 100 * float(np.mean(f1s))

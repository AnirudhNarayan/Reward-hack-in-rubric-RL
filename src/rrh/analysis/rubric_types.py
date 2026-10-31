"""Rubric-item categorization and per-category satisfaction (Sec. 4.2, Tables 2, 13, 14, 20, 21)."""

from __future__ import annotations

import asyncio
from collections import defaultdict

import numpy as np

from rrh.grading.clients import ChatClient
from rrh.grading.parsing import extract_json
from rrh.grading.prompts import (RUBRIC_CATEGORIES, RUBRIC_CATEGORY_SYSTEM_PROMPT,
                                 RUBRIC_CATEGORY_USER_PROMPT, format_conversation,
                                 format_rubric_list)
from rrh.grading.reward import item_satisfied

CATS = list(RUBRIC_CATEGORIES)
CAT_TYPE = {k: t for k, (t, _) in RUBRIC_CATEGORIES.items()}


async def classify_rubric(client: ChatClient, row: dict) -> list[str]:
    user = RUBRIC_CATEGORY_USER_PROMPT.format(
        question=format_conversation(row["prompt"]),
        rubric_list_string=format_rubric_list(row["criteria"]))
    try:
        obj = extract_json(await client.chat(RUBRIC_CATEGORY_SYSTEM_PROMPT, user))
    except ValueError:
        obj = {}
    labels = []
    for i in range(1, len(row["criteria"]) + 1):
        lab = str(obj.get(str(i), "other")).strip().lower()
        labels.append(lab if lab in CATS else "other")
    return labels


async def classify_all(client: ChatClient, rows: list[dict]) -> dict[str, list[str]]:
    labs = await asyncio.gather(*[classify_rubric(client, r) for r in rows])
    return {r["id"]: l for r, l in zip(rows, labs)}


def weight_shares(rows: list[dict], labels: dict[str, list[str]]) -> dict[str, float]:
    """Share of total |weight| per category and per type (Table 13)."""
    tot = defaultdict(float)
    for r in rows:
        for w, c in zip(r["weights"], labels[r["id"]]):
            tot[c] += abs(w)
    s = sum(tot.values())
    out = {c: tot[c] / s for c in CATS}
    for t in ("presence", "absence", "other"):
        out[f"{t}_total"] = sum(v for c, v in out.items() if CAT_TYPE.get(c) == t)
    return out


def per_prompt_satisfaction(row: dict, labels: list[str], met: list[bool | None]) -> dict:
    """Weight-averaged satisfaction per category / type for one graded response (NaN if absent).

    ``met`` may be fractional judge agreement (Table 14 uses fractional-judge satisfaction);
    pass the mean over judges of the per-item satisfied indicator.
    """
    num, den = defaultdict(float), defaultdict(float)
    for w, c, m in zip(row["weights"], labels, met):
        s = float(m)
        for key in (c, CAT_TYPE[c] + "_total", "total"):
            num[key] += abs(w) * s
            den[key] += abs(w)
    keys = CATS + ["presence_total", "absence_total", "other_total", "total"]
    return {k: (num[k] / den[k] if den[k] else float("nan")) for k in keys}


def fractional_satisfaction(row: dict, panel_met: list[list[bool | None]]) -> list[float]:
    """Mean over judges of item_satisfied (flips negative-weight items, App. I.5)."""
    return [float(np.mean([item_satisfied(w, jm[k]) for jm in panel_met]))
            for k, w in enumerate(row["weights"])]

"""Rubric-free pairwise judging with position flipping (Sec. 4.1, App. I.1-I.2).

Each (base, ckpt) pair is judged in both orders by each judge; per-response scores are
averaged over the two orders. A judge's winner is decided by the overall score (tie if equal).
The rubric-based winner for the same pair compares R under that judge's rubric grades.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Sequence

import numpy as np

from rrh.grading.clients import ChatClient
from rrh.grading.parsing import extract_json
from rrh.grading.prompts import (PAIRWISE_DIMENSIONS, RUBRIC_FREE_USER_PROMPT,
                                 format_conversation, rubric_free_system_prompt)

DIMS = PAIRWISE_DIMENSIONS + ["overall"]


def _scores(obj: dict, key: str) -> dict[str, float] | None:
    blk = obj.get(key) or obj.get(key.replace("response_", "@response_"))
    if not isinstance(blk, dict):
        return None
    out = {}
    for d in DIMS:
        v = blk.get(d)
        v = v.get("score") if isinstance(v, dict) else v
        try:
            out[d] = float(v)
        except (TypeError, ValueError):
            return None
    return out


async def judge_pair(client: ChatClient, domain: str, question, base: str, ckpt: str,
                     max_retries: int = 2) -> dict | None:
    """-> {"base": {dim: score}, "ckpt": {dim: score}} averaged over both orders."""
    system = rubric_free_system_prompt(domain)
    q = format_conversation(question)
    acc = {"base": [], "ckpt": []}
    for a_is_base in (True, False):
        ra, rb = (base, ckpt) if a_is_base else (ckpt, base)
        user = RUBRIC_FREE_USER_PROMPT.format(question=q, response_a=ra, response_b=rb)
        for attempt in range(max_retries + 1):
            try:
                obj = extract_json(await client.chat(system, user, sample_idx=attempt))
                sa, sb = _scores(obj, "response_A"), _scores(obj, "response_B")
                if sa and sb:
                    break
            except ValueError:
                pass
        else:
            return None
        acc["base"].append(sa if a_is_base else sb)
        acc["ckpt"].append(sb if a_is_base else sa)
    return {k: {d: float(np.mean([s[d] for s in v])) for d in DIMS} for k, v in acc.items()}


async def judge_all(client: ChatClient, domain: str, pairs: list[dict]) -> list[dict]:
    """pairs: {"id", "question", "base", "ckpt"}"""
    res = await asyncio.gather(*[judge_pair(client, domain, p["question"], p["base"], p["ckpt"])
                                 for p in pairs])
    return [{"id": p["id"], "judge": client.spec.name, **r} for p, r in zip(pairs, res) if r]


def winner(base_score: float, ckpt_score: float) -> str:
    return "ckpt" if ckpt_score > base_score else "base" if base_score > ckpt_score else "tie"


def aggregate_winner(votes: Sequence[str], rule: str) -> str | None:
    """Panel winner from per-judge winners. Pairs where any judge ties are excluded (Table 10)."""
    if "tie" in votes:
        return None
    c = Counter(votes)
    if rule == "consensus":
        return votes[0] if len(c) == 1 else None
    top, cnt = c.most_common(1)[0]
    return top if cnt * 2 > len(votes) else None


def dimension_summary(judged: list[dict]) -> dict:
    """Mean base / ckpt / delta per dimension (Table 11) and rubric-free ckpt win rate."""
    out = {}
    for d in DIMS:
        b = np.mean([r["base"][d] for r in judged])
        c = np.mean([r["ckpt"][d] for r in judged])
        out[d] = {"base": float(b), "ckpt": float(c), "delta": float(c - b),
                  "ckpt_win_rate": float(np.mean([r["ckpt"][d] > r["base"][d] for r in judged])),
                  "base_win_rate": float(np.mean([r["base"][d] > r["ckpt"][d] for r in judged]))}
    return out


def agreement_table(rubric_votes: dict[str, list[str]], free_votes: dict[str, list[str]],
                    rule: str) -> dict:
    """2x2 contingency of rubric-based vs rubric-free panel winners (Table 10)."""
    cells = Counter()
    for pid in rubric_votes.keys() & free_votes.keys():
        rw, fw = aggregate_winner(rubric_votes[pid], rule), aggregate_winner(free_votes[pid], rule)
        if rw and fw:
            cells[(rw, fw)] += 1
    n = sum(cells.values())
    agree = cells[("base", "base")] + cells[("ckpt", "ckpt")]
    return {"n": n, "agreement": agree / n if n else float("nan"),
            "cells": {f"rubric:{a}|free:{b}": cells[(a, b)]
                      for a in ("base", "ckpt") for b in ("base", "ckpt")}}

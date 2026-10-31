"""Claim extraction and within-prompt fixed-effects correlations (App. I.3-I.4, Tables 15, 16, 22)."""

from __future__ import annotations

import asyncio

import numpy as np
from scipy import stats

from rrh.grading.clients import ChatClient
from rrh.grading.parsing import extract_json
from rrh.grading.prompts import (CLAIM_EXTRACTION_SYSTEM_PROMPT, CLAIM_EXTRACTION_USER_PROMPT,
                                 format_conversation)


async def extract_claims(client: ChatClient, domain: str, question, response: str) -> dict:
    sys = CLAIM_EXTRACTION_SYSTEM_PROMPT.format(domain=domain)
    user = CLAIM_EXTRACTION_USER_PROMPT.format(question=format_conversation(question),
                                               response=response)
    try:
        claims = extract_json(await client.chat(sys, user)).get("claims", [])
    except (ValueError, AttributeError):
        return {"total_claims": float("nan"), "incorrect_claims": float("nan"),
                "error_rate": float("nan")}
    total = len(claims)
    bad = sum(1 for c in claims if isinstance(c, dict) and c.get("correct") is False)
    return {"total_claims": total, "incorrect_claims": bad,
            "error_rate": bad / total if total else float("nan")}


async def extract_all(client: ChatClient, domain: str, items: list[dict]) -> list[dict]:
    """items: {"id", "step", "question", "response"}"""
    res = await asyncio.gather(*[extract_claims(client, domain, it["question"], it["response"])
                                 for it in items])
    return [{"id": it["id"], "step": it["step"], **r} for it, r in zip(items, res)]


def demean_within_prompt(ids: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x~_{i,t} = x_{i,t} - mean_t x_{i,t}, NaN-safe (prompts missing x are excluded)."""
    out = np.full_like(x, np.nan, dtype=float)
    for pid in np.unique(ids):
        m = ids == pid
        v = x[m]
        if np.isfinite(v).any():
            out[m] = v - np.nanmean(v)
    return out


def fixed_effects_corr(ids, x, y) -> dict:
    """Pearson r between within-prompt demeaned x and y (pairs with NaN dropped)."""
    ids, x, y = np.asarray(ids), np.asarray(x, float), np.asarray(y, float)
    xd, yd = demean_within_prompt(ids, x), demean_within_prompt(ids, y)
    ok = np.isfinite(xd) & np.isfinite(yd)
    if ok.sum() < 3:
        return {"r": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    r, p = stats.pearsonr(xd[ok], yd[ok])
    return {"r": float(r), "p": float(p), "n": int(ok.sum())}

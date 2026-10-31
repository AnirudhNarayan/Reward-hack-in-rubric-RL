"""Rubric grader: one call grades ALL rubric items of a (prompt, response) pair (App. A.1)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

from rrh.grading.clients import ChatClient
from rrh.grading.parsing import parse_grader_output
from rrh.grading.prompts import (GRADER_SYSTEM_PROMPT, GRADER_USER_PROMPT, format_conversation,
                                 format_rubric_list)

log = logging.getLogger(__name__)


@dataclass
class GradeResult:
    met: list[bool | None]
    explanations: list[str]
    n_failed: int  # items still unparsed after all retries

    def to_dict(self) -> dict:
        return {"met": self.met, "explanations": self.explanations, "n_failed": self.n_failed}


class RubricGrader:
    def __init__(self, client: ChatClient, max_parse_retries: int = 3, show_points: bool = False):
        self.client = client
        self.max_parse_retries = max_parse_retries
        self.show_points = show_points

    def build_user_prompt(self, prompt, response: str, criteria: Sequence[str],
                          weights: Sequence[float] | None = None) -> str:
        show = self.show_points or (weights is not None and any(w < 0 for w in weights))
        return GRADER_USER_PROMPT.format(
            prompt=format_conversation(prompt), response=response,
            rubric_list_string=format_rubric_list(criteria, weights, show_points=show))

    async def grade(self, prompt, response: str, criteria: Sequence[str],
                    weights: Sequence[float] | None = None) -> GradeResult:
        n = len(criteria)
        user = self.build_user_prompt(prompt, response, criteria, weights)
        met: list[bool | None] = [None] * n
        expl = [""] * n
        for attempt in range(self.max_parse_retries + 1):
            # attempt 0 may hit the cache; retries bypass it via a distinct sample_idx.
            text = await self.client.chat(GRADER_SYSTEM_PROMPT, user, sample_idx=attempt)
            try:
                m, e = parse_grader_output(text, n)
            except ValueError as err:
                log.warning("%s parse failure (attempt %d): %s", self.client.spec.name, attempt, err)
                continue
            for i in range(n):
                if met[i] is None and m[i] is not None:
                    met[i], expl[i] = m[i], e[i]
            if all(x is not None for x in met):
                break
        n_failed = sum(x is None for x in met)
        if n_failed:
            log.warning("%s: %d/%d items unparsed after retries", self.client.spec.name, n_failed, n)
        return GradeResult(met, expl, n_failed)

    async def grade_many(self, items: list[dict]) -> list[GradeResult]:
        """items: dicts with prompt, response, criteria, weights."""
        return await asyncio.gather(*[
            self.grade(it["prompt"], it["response"], it["criteria"], it.get("weights"))
            for it in items])

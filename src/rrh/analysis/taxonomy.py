"""Verifier failure-mode taxonomy (Sec. 3.2, App. H).

exploited instance -> one structural sentence ("The verifier failed because it ...",
extractor model) -> label in {A.1, A.2, B.1, B.2, C.1, C.2, OTHER} (classifier model).
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict

from rrh.grading.clients import ChatClient
from rrh.grading.parsing import extract_json
from rrh.grading.prompts import (FAILURE_EXTRACTION_SYSTEM_PROMPT, FAILURE_EXTRACTION_USER_PROMPT,
                                 TAXONOMY, TAXONOMY_CLASSIFIER_SYSTEM_PROMPT,
                                 TAXONOMY_CLASSIFIER_USER_PROMPT)

LABELS = list(TAXONOMY) + ["OTHER"]
PREFIX = "The verifier failed because it"


def _first_sentence(text: str) -> str:
    text = text.strip().strip('"')
    i = text.find(PREFIX)
    return text[i:] if i >= 0 else text


async def extract_sentence(client: ChatClient, inst: dict) -> str:
    pe = (inst["panel_explanations"] + ["", "", ""])[:3]
    user = FAILURE_EXTRACTION_USER_PROMPT.format(
        criterion=inst["criterion"], verifier_explanation=inst["verifier_explanation"],
        panel_explanation_1=pe[0], panel_explanation_2=pe[1], panel_explanation_3=pe[2])
    return _first_sentence(await client.chat(FAILURE_EXTRACTION_SYSTEM_PROMPT, user))


async def classify_sentence(client: ChatClient, sentence: str) -> str:
    text = await client.chat(TAXONOMY_CLASSIFIER_SYSTEM_PROMPT,
                             TAXONOMY_CLASSIFIER_USER_PROMPT.format(sentence=sentence))
    try:
        label = str(extract_json(text).get("label", "OTHER")).strip().upper()
    except (ValueError, AttributeError):
        label = "OTHER"
    return label if label in LABELS else "OTHER"


async def run_taxonomy(extractor: ChatClient, classifier: ChatClient,
                       instances: list[dict]) -> list[dict]:
    async def one(inst):
        s = await extract_sentence(extractor, inst)
        return {**inst, "failure_sentence": s, "label": await classify_sentence(classifier, s)}
    return await asyncio.gather(*[one(i) for i in instances])


def summarize(labeled: list[dict]) -> dict:
    """Parent/sub-mode shares and per-step counts (Figure 3).

    The paper's shares (A 36.0 / B 34.6 / C 29.4) sum to 100%, so shares are reported over
    classified cases (OTHER excluded); the OTHER rate is reported separately.
    """
    n = len(labeled)
    sub = Counter(r["label"] for r in labeled)
    n_cls = n - sub["OTHER"]
    parent = Counter(r["label"][0] for r in labeled if r["label"] != "OTHER")
    per_step: dict[int, Counter] = defaultdict(Counter)
    for r in labeled:
        per_step[r["step"]][r["label"]] += 1
    return {"n": n, "n_classified": n_cls, "other_rate": sub["OTHER"] / n if n else 0.0,
            "sub_share": {k: sub[k] / n_cls for k in TAXONOMY} if n_cls else {},
            "parent_share": {k: parent[k] / n_cls for k in "ABC"} if n_cls else {},
            "per_step_counts": {t: dict(c) for t, c in sorted(per_step.items())}}

"""ResearchQA (Yifei et al.; HF ``realliyifei/ResearchQA``) for RP-OPD.

The HF repo ships full.json (21,414), test.json (3,750) and valid.json (703). The paper's 16,961
training prompts are exactly full - test - valid; its 500 evaluation prompts are drawn from test
(D19). Rubric items are unweighted yes/no questions ("Does the response include ...?"); the
paper's App. A.1 card shows them as "+10 The response include ...", i.e. weight 10 and the
leading "Does the" rewritten to "The" -- reproduced by ``rubric_item_to_criterion``.
"""

from __future__ import annotations

import json
import re

HF_REPO = "realliyifei/ResearchQA"
ITEM_WEIGHT = 10.0
_DOES_RE = re.compile(r"^Does the response\b")


def rubric_item_to_criterion(item: str) -> str:
    s = item.strip()
    if _DOES_RE.match(s):
        s = "The response" + s[len("Does the response"):]
        if s.endswith("?"):
            s = s[:-1] + "."
    return s


def to_canonical(ex: dict) -> dict:
    crit = [rubric_item_to_criterion(r["rubric_item"]) for r in ex["rubric"]]
    return {"id": f"rqa-{ex['id']}", "domain": "science", "source": "researchqa",
            "field": ex.get("field"), "prompt": [{"role": "user", "content": ex["query"].strip()}],
            "criteria": crit, "weights": [ITEM_WEIGHT] * len(crit)}


def load_splits(cache_dir: str | None = None) -> dict[str, list[dict]]:
    from huggingface_hub import hf_hub_download
    raw = {}
    for s in ("full", "test", "valid"):
        with open(hf_hub_download(HF_REPO, f"{s}.json", repo_type="dataset", cache_dir=cache_dir),
                  encoding="utf-8") as f:
            raw[s] = json.load(f)
    held = {e["id"] for e in raw["test"]} | {e["id"] for e in raw["valid"]}
    train = [e for e in raw["full"] if e["id"] not in held]
    return {"train": [to_canonical(e) for e in train if e.get("rubric")],
            "test": [to_canonical(e) for e in raw["test"] if e.get("rubric")],
            "valid": [to_canonical(e) for e in raw["valid"] if e.get("rubric")]}

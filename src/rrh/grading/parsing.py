"""Robust JSON extraction from LLM grader output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Return the first JSON object in ``text`` (fenced ```json block preferred).

    Raises ValueError if no object can be parsed.
    """
    if text is None:
        raise ValueError("empty grader output")
    # gpt-oss style reasoning leaks: keep only the final channel if present.
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>")[-1]
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        start = cand.find("{")
        end = cand.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cand[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from grader output: {text[:200]!r}")


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "met", "1"}:
            return True
        if s in {"false", "no", "not_met", "not met", "0"}:
            return False
    return None


def parse_grader_output(text: str, n_items: int) -> tuple[list[bool | None], list[str]]:
    """Parse the A.1 grader JSON into per-item (criteria_met, explanation).

    Missing or malformed items come back as ``None`` so the caller can retry them.
    """
    obj = extract_json(text)
    if not isinstance(obj, dict):
        raise ValueError("grader JSON is not an object")
    met: list[bool | None] = [None] * n_items
    expl: list[str] = [""] * n_items
    for k, v in obj.items():
        try:
            idx = int(str(k).strip().rstrip(".")) - 1
        except ValueError:
            continue
        if not 0 <= idx < n_items:
            continue
        if isinstance(v, dict):
            met[idx] = _as_bool(v.get("criteria_met"))
            expl[idx] = str(v.get("explanation", ""))
        else:
            met[idx] = _as_bool(v)
    return met, expl

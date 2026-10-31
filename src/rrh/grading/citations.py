"""Citation checks for shortcut-aware grading (App. C / C.1).

The judge quotes response lines as evidence for content criteria. A script then removes credit
when the quoted text "is missing, is only a heading, or merely asks for an answer without
providing one", and (C.1) when a "line"-scoped quote is not an exact contiguous substring of the
cited numbered line. Whole-response evidence is not checked.
"""

from __future__ import annotations

import re
from typing import Sequence

_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.*|\*\*[^*]+\*\*:?|__[^_]+__:?|[A-Z][^.!?]{0,80}:)\s*$")
# OURS: imperative openers that only ask for content ("Discuss ...", "Provide ...") -- same family
# the paper's App. C.2 outline detector keys on.
_ASK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]*\*\*:?\s*)?"
                     r"(discuss|provide|explain|describe|include|mention|derive|outline|present|"
                     r"summarize|address|cite|elaborate)\b",
                     re.IGNORECASE)


def is_heading(quote: str) -> bool:
    return bool(_HEADING_RE.match(quote.strip()))


def only_asks_for_answer(quote: str) -> bool:
    return bool(_ASK_RE.match(quote.strip()))


def _norm_line_no(x) -> int | None:
    try:
        s = str(x).strip().upper().lstrip("L")
        return int(s)
    except (TypeError, ValueError):
        return None


def check_item_evidence(item: dict, lines: Sequence[str]) -> tuple[bool, str]:
    """-> (credit_kept, reason). Applies only to positive items the judge marked met; callers
    skip negative items (polarity is preserved, App. C.1)."""
    scope = str(item.get("evidence_scope", "line")).strip().lower()
    if scope == "whole_response":
        return True, "whole_response"
    ev = item.get("evidence") or []
    if isinstance(ev, dict):
        ev = [ev]
    quotes = []
    for e in ev:
        if isinstance(e, dict):
            quotes.append((_norm_line_no(e.get("line")), str(e.get("quote", ""))))
        elif isinstance(e, str):
            quotes.append((None, e))
    quotes = [(ln, q) for ln, q in quotes if q.strip()]
    if not quotes:
        return False, "missing_quote"
    for ln, q in quotes:
        if ln is None or not 1 <= ln <= len(lines):
            return False, "bad_line_number"
        if q not in lines[ln - 1]:
            return False, "quote_not_in_line"
    if all(is_heading(q) for _, q in quotes):
        return False, "heading_only"
    if all(only_asks_for_answer(q) for _, q in quotes):
        return False, "asks_for_answer"
    return True, "ok"

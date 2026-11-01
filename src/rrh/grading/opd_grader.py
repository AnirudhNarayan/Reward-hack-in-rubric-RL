"""Rubric judges for the RP-OPD replication + resumable grading of a run's generations.

* ``BooleanRubricJudge``: App. A.1 prompt (Qwen3-32B in the paper -> Llama-3.3-70B-Instruct),
  one call grades every criterion; JSON {"0": true, ...}; score = Eq. (1).
* ``ShortcutAwareJudge``: App. C.1 (GPT-4o-mini -> Gemma-3-27B-it): standard_met and
  hack_aware_met per item + response-level classification, then the citation checks. Reports
  three scores, as in Table 4: standard, LLM regrade before evidence checks, after evidence checks
  (= the "shortcut-aware" score used in Table 1).

    python -m rrh.grading.opd_grader --run_dir results/rp_opd/<run> \
        --judge llama-3.3-70b-judge --mode boolean
    python -m rrh.grading.opd_grader --run_dir results/rp_opd/<run> \
        --judge gemma-3-27b-judge --mode shortcut

Raw judge outputs go to judgments/<judge>[-shortcut]/<evalset>/<step>.jsonl (append-only,
skips already-graded rows) and every call is cached in the SQLite response cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from rrh.grading.citations import check_item_evidence
from rrh.grading.opd_prompts import (SHORTCUT_AWARE_SYSTEM_PROMPT, number_lines,
                                     rubric_judge_prompt, shortcut_aware_user_prompt)
from rrh.grading.parsing import _as_bool, extract_json
from rrh.grading.reward import normalized_rubric_score
from rrh.opd.contexts import conversation
from rrh.results.layout import RunDir, step_name
from rrh.utils.io import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------- parsing
def parse_boolean_json(text: str, n: int) -> list[bool | None]:
    obj = extract_json(text)
    if not isinstance(obj, dict):
        raise ValueError("judge JSON is not an object")
    met: list[bool | None] = [None] * n
    for k, v in obj.items():
        try:
            i = int(str(k).strip().rstrip("."))
        except ValueError:
            continue
        if 0 <= i < n:
            if isinstance(v, dict):
                v = v.get("met", v.get("criteria_met"))
            met[i] = _as_bool(v)
    return met


def parse_shortcut_json(text: str, n: int) -> dict:
    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        raise ValueError("shortcut-aware JSON lacks an 'items' list")
    items: dict[int, dict] = {}
    for it in obj["items"]:
        if not isinstance(it, dict):
            continue
        try:
            i = int(it.get("rubric_idx"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and _as_bool(it.get("standard_met")) is not None:
            items[i] = it
    affected = set()
    for x in obj.get("affected_rubric_indices") or []:
        try:
            affected.add(int(x))
        except (TypeError, ValueError):
            pass
    return {"classification": str(obj.get("response_classification", "none")).strip().lower(),
            "confidence": str(obj.get("confidence", "low")).strip().lower(),
            "affected": affected, "items": items}


def combine_shortcut(parsed: dict, response: str, weights: Sequence[float]) -> dict:
    """Apply the C.1 gating rule and the citation checks. Polarity of negative items unchanged."""
    lines = number_lines(response)
    gate = parsed["classification"] == "confirmed" and parsed["confidence"] == "high"
    std, pre, final, reasons = [], [], [], []
    for i, w in enumerate(weights):
        it = parsed["items"].get(i)
        if it is None:
            std.append(None), pre.append(None), final.append(None), reasons.append("missing")
            continue
        s = _as_bool(it.get("standard_met"))
        h = _as_bool(it.get("hack_aware_met"))
        p = s
        reason = "standard"
        if w > 0 and s and h is False and gate and i in parsed["affected"]:
            p, reason = False, "shortcut_removed"
        f = p
        if w > 0 and p:
            kept, why = check_item_evidence(it, lines)
            if not kept:
                f, reason = False, f"citation:{why}"
        std.append(s), pre.append(p), final.append(f), reasons.append(reason)
    return {"standard_met": std, "hack_aware_pre_citation": pre, "met": final, "reasons": reasons,
            "score_standard": normalized_rubric_score(weights, std),
            "score_pre_citation": normalized_rubric_score(weights, pre),
            "score": normalized_rubric_score(weights, final),
            "classification": parsed["classification"], "confidence": parsed["confidence"]}


# --------------------------------------------------------------------------------------- judges
class BooleanRubricJudge:
    mode = "boolean"

    def __init__(self, client, max_parse_retries: int = 3):
        self.client = client
        self.retries = max_parse_retries

    async def grade(self, messages, response: str, criteria, weights) -> dict:
        n = len(criteria)
        prompt = rubric_judge_prompt(messages, response, criteria, weights)
        met: list[bool | None] = [None] * n
        raws = []
        for attempt in range(self.retries + 1):
            text = await self.client.chat(None, prompt, sample_idx=attempt)
            raws.append(text)
            try:
                m = parse_boolean_json(text, n)
            except ValueError:
                continue
            met = [a if a is not None else b for a, b in zip(met, m)]
            if all(x is not None for x in met):
                break
        return {"met": met, "n_failed": sum(x is None for x in met), "raw": raws,
                "score": normalized_rubric_score(weights, met)}


class ShortcutAwareJudge:
    mode = "shortcut"

    def __init__(self, client, max_parse_retries: int = 3):
        self.client = client
        self.retries = max_parse_retries

    async def grade(self, messages, response: str, criteria, weights) -> dict:
        n = len(criteria)
        user = shortcut_aware_user_prompt(messages, response, criteria, weights)
        raws, parsed = [], None
        for attempt in range(self.retries + 1):
            text = await self.client.chat(SHORTCUT_AWARE_SYSTEM_PROMPT, user, sample_idx=attempt)
            raws.append(text)
            try:
                p = parse_shortcut_json(text, n)
            except ValueError:
                continue
            if parsed is None:
                parsed = p
            else:   # fill items missing from the earlier parse
                for i, it in p["items"].items():
                    parsed["items"].setdefault(i, it)
            if len(parsed["items"]) == n:
                break
        if parsed is None:
            parsed = {"classification": "none", "confidence": "low", "affected": set(), "items": {}}
        out = combine_shortcut(parsed, response, weights)
        out.update({"n_failed": sum(x is None for x in out["met"]), "raw": raws})
        return out


# ------------------------------------------------------------------------------- run grading
def judge_dir_name(judge: str, mode: str) -> str:
    return judge if mode == "boolean" else f"{judge}-shortcut"


def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, seed: int = 0) -> list[float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(judged: list[dict], gens: dict, meta: dict) -> dict:
    scores = [j["score"] for j in judged]
    out = dict(meta)
    out.update({"n": len(judged), "score_mean": float(np.mean(scores)) if scores else float("nan"),
                "score_ci95": bootstrap_ci(scores),
                "n_failed_items": int(sum(j.get("n_failed", 0) for j in judged))})
    for k in ("score_standard", "score_pre_citation"):
        vals = [j[k] for j in judged if k in j]
        if vals:
            out[k + "_mean"] = float(np.mean(vals))
    g = [gens[(j["id"], j.get("sample_idx", 0))] for j in judged if (j["id"], j.get("sample_idx", 0)) in gens]
    if g:
        out["mean_tokens"] = float(np.mean([x["n_tokens"] for x in g]))
        ent = [x.get("entropy_top20") for x in g if x.get("entropy_top20") == x.get("entropy_top20")]
        out["entropy_top20_mean"] = float(np.mean(ent)) if ent else float("nan")
        out["truncated_frac"] = float(np.mean([x.get("finish_reason") == "length" for x in g]))
    return out


async def grade_file(judge, gen_path: Path, out_path: Path, data: dict[str, dict],
                     subset: set[str] | None, chunk: int = 256) -> list[dict]:
    gens = [g for g in read_jsonl(gen_path) if subset is None or g["id"] in subset]
    done = {}
    if out_path.exists():
        for j in read_jsonl(out_path):
            done[(j["id"], j.get("sample_idx", 0))] = j
    todo = [g for g in gens if (g["id"], g.get("sample_idx", 0)) not in done]
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]

        async def one(g):
            row = data[g["id"]]
            r = await judge.grade(conversation(row), g["response"], row["criteria"], row["weights"])
            r.update({"id": g["id"], "sample_idx": g.get("sample_idx", 0), "step": g["step"],
                      "judge": judge.client.spec.name, "mode": judge.mode})
            return r

        res = await asyncio.gather(*[one(g) for g in part])
        write_jsonl(out_path, res, append=True)
        for r in res:
            done[(r["id"], r["sample_idx"])] = r
    return [done[(g["id"], g.get("sample_idx", 0))] for g in gens]


def subset_ids(rows: list[dict], n: int | None) -> set[str] | None:
    if not n:
        return None
    import hashlib
    ids = sorted((r["id"] for r in rows), key=lambda i: hashlib.sha1(i.encode()).hexdigest())
    return set(ids[:n])


async def grade_run(run: RunDir, judge_alias: str, mode: str, evalsets: list[str] | None,
                    steps: list[str] | None, subset_n: int | None, runtime) -> None:
    cfg = run.read_config()
    client = runtime.client(judge_alias)
    judge = BooleanRubricJudge(client) if mode == "boolean" else ShortcutAwareJudge(client)
    jname = judge_dir_name(judge_alias, mode)
    if steps:   # "final" = the last training step (the Table 1 checkpoint, D21)
        steps = [step_name(cfg["train"]["total_steps"]) if s == "final" else s for s in steps]
    for evalset, path in cfg["eval"]["eval_sets"].items():
        if evalsets and evalset not in evalsets:
            continue
        rows = read_jsonl(path)
        data = {r["id"]: r for r in rows}
        subset = subset_ids(rows, subset_n)
        gdir = run.path / "generations" / evalset
        if not gdir.exists():
            continue
        for gp in sorted(gdir.glob("*.jsonl")):
            step = gp.stem
            if steps and step not in steps and step.replace("step_", "").lstrip("0") not in steps:
                continue
            out = run.judgments(jname, evalset, step)
            judged = await grade_file(judge, gp, out, data, subset)
            gens = {(g["id"], g.get("sample_idx", 0)): g for g in read_jsonl(gp)}
            meta = {"run_id": run.run_id, "evalset": evalset, "judge": judge_alias, "mode": mode,
                    "step": step, "subset_n": subset_n}
            summ = summarize(judged, gens, meta)
            mp = run.eval_metrics(evalset, jname, step)
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps(summ, indent=2))
            print(f"[grade] {jname} {evalset} {step}: n={summ['n']} score={summ['score_mean']:.2f}",
                  flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--judge", required=True, help="alias in configs/models.yaml")
    ap.add_argument("--mode", choices=["boolean", "shortcut"], default="boolean")
    ap.add_argument("--evalsets", nargs="*", default=None)
    ap.add_argument("--steps", nargs="*", default=None, help="e.g. step_000100 teacher 980")
    ap.add_argument("--subset_n", type=int, default=None, help="grade a fixed hashed subset of ids")
    ap.add_argument("--cache", default=os.environ.get("RRH_CACHE"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from rrh.utils.runtime import Runtime
    rt = Runtime(cache_path=a.cache)
    run = RunDir(a.run_dir)
    asyncio.run(grade_run(run, a.judge, a.mode, a.evalsets, a.steps, a.subset_n, rt))
    cost = rt.cost.summary()
    print(cost, flush=True)
    with open(run.path / "metrics" / "judge_cost.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"judge": a.judge, "mode": a.mode, "usage": json.loads(cost)}) + "\n")


if __name__ == "__main__":
    main()

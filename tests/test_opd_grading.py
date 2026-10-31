"""Judges (A.1 boolean, C.1 shortcut-aware), citation checks, resumable grading, report, agreement."""

import asyncio
import json

import pytest

from rrh.grading.citations import check_item_evidence
from rrh.grading.clients import ModelSpec
from rrh.grading.opd_grader import (BooleanRubricJudge, ShortcutAwareJudge, combine_shortcut,
                                    grade_run, parse_boolean_json, parse_shortcut_json)
from rrh.results.layout import create_run, read_index
from rrh.utils.io import read_jsonl, write_jsonl


class FakeClient:
    def __init__(self, reply_fn, name="fake"):
        self.reply_fn = reply_fn
        self.calls = []
        self.spec = ModelSpec(name=name, provider="vllm", model=name)

    async def chat(self, system, user, use_cache=True, sample_idx=0):
        self.calls.append((system, user, sample_idx))
        return self.reply_fn(system, user, sample_idx)


class FakeRuntime:
    def __init__(self, clients):
        self.clients = clients

    def client(self, name):
        return self.clients[name]


def test_parse_boolean_zero_indexed():
    assert parse_boolean_json('```json\n{"0": true, "1": "false", "5": true}\n```', 3) == [True, False, None]
    with pytest.raises(ValueError):
        parse_boolean_json("nope", 2)


def test_boolean_judge_retries_and_scores_eq1():
    replies = iter(['{"0": true}', '{"0": false, "1": true, "2": false}'])
    c = FakeClient(lambda s, u, i: next(replies))
    res = asyncio.run(BooleanRubricJudge(c).grade([{"role": "user", "content": "Q"}], "A",
                                                  ["a", "b", "c"], [5, 5, -10]))
    assert res["met"] == [True, True, False] and res["n_failed"] == 0
    assert res["score"] == pytest.approx(1.0)
    assert c.calls[0][0] is None and "0. [5 points] a" in c.calls[0][1]
    assert [x[2] for x in c.calls] == [0, 1]


def _sc_json(items, cls="confirmed", conf="high", affected=(0,)):
    return json.dumps({"response_classification": cls, "confidence": conf,
                       "affected_rubric_indices": list(affected), "items": items})


RESP = "## Derivation\nZ = sum_i exp(-beta E_i)\nDiscuss the partition function in detail.\nThe response meets the rubric."


def test_citation_checks():
    lines = RESP.split("\n")
    ok = {"evidence_scope": "line", "evidence": [{"line": 2, "quote": "Z = sum_i exp(-beta E_i)"}]}
    assert check_item_evidence(ok, lines) == (True, "ok")
    assert check_item_evidence({"evidence_scope": "line", "evidence": []}, lines)[1] == "missing_quote"
    assert check_item_evidence({"evidence_scope": "line", "evidence": [{"line": 3, "quote": "Z = sum"}]},
                               lines)[1] == "quote_not_in_line"
    assert check_item_evidence({"evidence": [{"line": "L1", "quote": "## Derivation"}]}, lines)[1] == "heading_only"
    assert check_item_evidence({"evidence": [{"line": 3, "quote": "Discuss the partition function"}]},
                               lines)[1] == "asks_for_answer"
    assert check_item_evidence({"evidence_scope": "whole_response", "evidence": []}, lines)[0]


def test_shortcut_gating_and_polarity():
    items = [
        # positive, shortcut-only -> removed (confirmed/high/listed)
        {"rubric_idx": 0, "standard_met": True, "hack_aware_met": False, "evidence_scope": "line",
         "evidence": [{"line": 4, "quote": "The response meets the rubric."}]},
        # positive, genuine evidence -> kept
        {"rubric_idx": 1, "standard_met": True, "hack_aware_met": True, "evidence_scope": "line",
         "evidence": [{"line": 2, "quote": "Z = sum_i exp(-beta E_i)"}]},
        # positive, hack-aware false but NOT listed as affected -> stays; but citation is a heading
        {"rubric_idx": 2, "standard_met": True, "hack_aware_met": False, "evidence_scope": "line",
         "evidence": [{"line": 1, "quote": "## Derivation"}]},
        # negative pitfall present: polarity unchanged even if hack_aware says otherwise
        {"rubric_idx": 3, "standard_met": True, "hack_aware_met": False, "evidence_scope": "whole_response",
         "evidence": []},
    ]
    w = [4, 4, 2, -5]
    out = combine_shortcut(parse_shortcut_json(_sc_json(items), 4), RESP, w)
    assert out["standard_met"] == [True, True, True, True]
    assert out["hack_aware_pre_citation"] == [False, True, True, True]
    assert out["met"] == [False, True, False, True]
    assert out["score_standard"] == pytest.approx((4 + 4 + 2 - 5) / 10)
    assert out["score_pre_citation"] == pytest.approx((4 + 2 - 5) / 10)
    assert out["score"] == pytest.approx((4 - 5) / 10)
    # not confirmed -> no shortcut removal (citation checks still apply)
    amb = combine_shortcut(parse_shortcut_json(_sc_json(items, cls="ambiguous"), 4), RESP, w)
    assert amb["hack_aware_pre_citation"][0] is True


def test_shortcut_judge_end_to_end():
    items = [{"rubric_idx": 0, "standard_met": True, "hack_aware_met": True, "evidence_scope": "whole_response",
              "evidence": []}]
    c = FakeClient(lambda s, u, i: _sc_json(items, cls="none"))
    res = asyncio.run(ShortcutAwareJudge(c).grade([{"role": "user", "content": "Q"}], "A", ["a"], [1]))
    assert res["score"] == 1.0 and res["n_failed"] == 0
    assert c.calls[0][0].startswith("You are an independent rubric evaluator.")
    assert "L1: A" in c.calls[0][1]


@pytest.fixture
def synthetic_run(tmp_path):
    root = tmp_path / "results"
    rd = create_run(root, "rp_opd", "llama-3.1-8b-instruct", "strong", "healthbench", 0, run_id="r1")
    evp = tmp_path / "eval.jsonl"
    rows = [{"id": f"p{i}", "prompt": [{"role": "user", "content": f"q{i}"}],
             "criteria": ["c0", "c1"], "weights": [3.0, -1.0]} for i in range(6)]
    write_jsonl(evp, rows)
    rd.write_config({"seed": 0, "student": {"name": "llama-3.1-8b-instruct", "path": "x"},
                     "teacher": {"rubric_in_context": True}, "data": {"name": "healthbench"},
                     "eval": {"curve_judge": "strong", "independent_judge": "weak",
                              "eval_sets": {"heldout": str(evp)}},
                     "train": {"global_batch": 512, "total_steps": 2}, "loss": {"topk": 256}})
    for step, text in [(0, "short"), (20, "better answer"), ("teacher", "teacher answer")]:
        write_jsonl(rd.generations("heldout", step),
                    [{"id": r["id"], "step": step, "evalset": "heldout", "sample_idx": 0,
                      "response": text, "n_tokens": len(text), "finish_reason": "stop",
                      "entropy_top20": 1.0} for r in rows])
    return rd


def test_grade_run_report_and_agreement(synthetic_run, tmp_path):
    from rrh.analysis import judge_agreement, opd_report
    rd = synthetic_run

    def strong(s, u, i):   # met c0 only for longer answers; never the pitfall
        good = "better answer" in u or "teacher answer" in u
        return json.dumps({"0": good, "1": False})

    def weak(s, u, i):      # always credits c0, sometimes flags the pitfall
        return json.dumps({"0": True, "1": "q0" in u})

    def weak_sc(s, u, i):
        return _sc_json([{"rubric_idx": 0, "standard_met": True, "hack_aware_met": True,
                          "evidence_scope": "whole_response", "evidence": []},
                         {"rubric_idx": 1, "standard_met": False, "hack_aware_met": False,
                          "evidence_scope": "whole_response", "evidence": []}], cls="none")

    calls = {"strong": FakeClient(strong, "strong"), "weak": FakeClient(weak, "weak"),
             "weak_sc": FakeClient(weak_sc, "weak")}
    rt = FakeRuntime({"strong": calls["strong"], "weak": calls["weak"]})
    asyncio.run(grade_run(rd, "strong", "boolean", None, None, None, rt))
    asyncio.run(grade_run(rd, "weak", "boolean", None, None, None, rt))
    n_before = len(calls["strong"].calls)
    asyncio.run(grade_run(rd, "strong", "boolean", None, None, None, rt))     # resumable: no new calls
    assert len(calls["strong"].calls) == n_before
    asyncio.run(grade_run(rd, "weak", "shortcut", None, None, None, FakeRuntime({"weak": calls["weak_sc"]})))

    m0 = json.loads(rd.eval_metrics("heldout", "strong", 0).read_text())
    m20 = json.loads(rd.eval_metrics("heldout", "strong", 20).read_text())
    assert m0["score_mean"] == 0.0 and m20["score_mean"] == 1.0 and m20["n"] == 6
    assert json.loads(rd.eval_metrics("heldout", "weak-shortcut", 20).read_text())["score_mean"] == 1.0
    assert len(read_jsonl(rd.judgments("strong", "heldout", "teacher"))) == 6

    s = opd_report.run_report(rd)
    assert s["heldout"]["best_step"] == 20 and s["heldout"]["teacher_score"] == 1.0
    assert s["heldout"]["final_shortcut_aware"] == 1.0
    assert (rd.path / "figures" / "fig2_heldout.png").exists()
    assert (rd.path / "tables" / "eval_heldout.csv").exists()
    assert json.loads((rd.path / "stage2_handoff.json").read_text())["run_id"] == "r1"
    idx = read_index(rd.results_root)
    assert idx[0]["best_step"] == "20" and idx[0]["teacher_ref_score"] == "1.0"

    rows = judge_agreement.compare([rd], "strong", ["weak", "weak-shortcut"])
    weak_row = rows[0]
    # 18 responses x 2 criteria: c0 disagrees on the 6 step-0 responses (weak FP), c1 disagrees on q0 x 3
    assert weak_row["n_criteria"] == 36
    assert weak_row["all_fp"] == pytest.approx(100 * 9 / 36)
    assert weak_row["all_fn"] == 0.0
    judge_agreement.write_outputs(rows, tmp_path / "jv")
    assert (tmp_path / "jv" / "tables" / "judge_agreement.csv").exists()

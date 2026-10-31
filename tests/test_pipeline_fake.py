"""End-to-end grading path with a fake LLM client (no network)."""

import asyncio
import json

from rrh.analysis.pairwise import judge_pair
from rrh.analysis.taxonomy import classify_sentence, extract_sentence
from rrh.grading.clients import ModelSpec
from rrh.grading.grader import RubricGrader
from rrh.grading.prompts import RUBRIC_FREE_SYSTEM_PROMPT_MEDICAL, rubric_free_system_prompt


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.spec = ModelSpec(name="fake", provider="openai", model="fake")

    async def chat(self, system, user, use_cache=True, sample_idx=0):
        self.calls.append((system, user, sample_idx))
        return self.replies.pop(0)


def test_grader_retries_missing_items():
    first = '```json\n{"1": {"explanation": "a", "criteria_met": true}}\n```'
    second = '```json\n{"1": {"explanation": "x", "criteria_met": false},' \
             ' "2": {"explanation": "b", "criteria_met": false}}\n```'
    c = FakeClient([first, second])
    res = asyncio.run(RubricGrader(c).grade("Q?", "A.", ["c1", "c2"], [1, 1]))
    assert res.met == [True, False]          # item 1 kept from the first parse
    assert res.n_failed == 0
    assert [s for _, _, s in c.calls] == [0, 1]
    assert "<rubrics>\n1. c1\n2. c2\n</rubrics>" in c.calls[0][1]


def test_grader_shows_points_for_negative_items():
    c = FakeClient(['{"1": {"criteria_met": true}, "2": {"criteria_met": false}}'])
    asyncio.run(RubricGrader(c).grade([{"role": "user", "content": "Q"}], "A", ["c1", "c2"], [5, -3]))
    assert "2. [-3 points] c2" in c.calls[0][1]
    assert "user: Q" in c.calls[0][1]


def _pair_json(a_overall, b_overall):
    blk = lambda o: {d: {"score": 4} for d in ["completeness", "factual_correctness", "conciseness",
                                                "relevance", "safety"]} | {"overall": {"score": o}}
    return json.dumps({"response_A": blk(a_overall), "response_B": blk(b_overall)})


def test_pairwise_position_flip_average():
    # order 1: A=base gets 6, B=ckpt 2; order 2: A=ckpt gets 4, B=base 5
    c = FakeClient([_pair_json(6, 2), _pair_json(4, 5)])
    r = asyncio.run(judge_pair(c, "medical", "Q", "BASE", "CKPT"))
    assert r["base"]["overall"] == 5.5 and r["ckpt"]["overall"] == 3.0
    assert "<@response_A>\nBASE" in c.calls[0][1] and "<@response_A>\nCKPT" in c.calls[1][1]
    assert c.calls[0][0] == RUBRIC_FREE_SYSTEM_PROMPT_MEDICAL


def test_science_judge_prompt_is_domain_swapped():
    s = rubric_free_system_prompt("science")
    assert "medical" not in s.lower()


def test_taxonomy_flow():
    ext = FakeClient(['Sure. The verifier failed because it matched on topic.'])
    s = asyncio.run(extract_sentence(ext, {"criterion": "c", "verifier_explanation": "v",
                                           "panel_explanations": ["p1", "p2", "p3"]}))
    assert s == "The verifier failed because it matched on topic."
    cls = FakeClient(['{"label": "C.2"}'])
    assert asyncio.run(classify_sentence(cls, s)) == "C.2"
    assert asyncio.run(classify_sentence(FakeClient(['{"label": "Z.9"}']), s)) == "OTHER"

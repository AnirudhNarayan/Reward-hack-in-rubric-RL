import math

import numpy as np
import pytest

from rrh.analysis.claims import demean_within_prompt, fixed_effects_corr
from rrh.analysis.exploitation import (exploitation_rate, exploited_instances, index_judgments,
                                       per_prompt_terms, reward_curves)
from rrh.analysis.pairwise import aggregate_winner
from rrh.analysis.self_gap import compare_with_rewards, self_gap_by_step
from rrh.analysis.verifier_agreement import agreement_stats, candidate_vs_panel, macro_f1
from rrh.grading.parsing import extract_json, parse_grader_output
from rrh.grading.prompts import (GRADER_SYSTEM_PROMPT, GRADER_USER_PROMPT, format_rubric_list,
                                 rubric_conditioned_system_prompt)
from rrh.grading.reward import majority_vote, rubric_reward, unanimous_consensus, unanimous_reject


# ---------------- reward ----------------
def test_reward_positive_only():
    assert rubric_reward([1, 2, 3], [True, False, True]) == pytest.approx(4 / 6)


def test_reward_with_negative_items():
    # positive met (3) + negative avoided (|−2|) over 3+2+1
    assert rubric_reward([3, -2, 1], [True, False, False]) == pytest.approx(5 / 6)
    assert rubric_reward([3, -2, 1], [True, True, True]) == pytest.approx(4 / 6)


def test_reward_bounds_and_none():
    assert rubric_reward([1, 1], [None, None]) == 0.0
    assert rubric_reward([], []) == 0.0


def test_consensus_rules():
    panel = [[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]]
    assert unanimous_consensus(panel) == [True, False, False, False]
    assert unanimous_consensus(panel, "not_all_reject") == [True, True, False, True]
    assert unanimous_reject(panel) == [False, False, True, False]
    assert majority_vote(panel) == [True, True, False, True]


# ---------------- parsing ----------------
def test_parse_fenced_json():
    txt = '```json\n{"1": {"explanation": "ok", "criteria_met": true},' \
          ' "2": {"explanation": "no", "criteria_met": false}}\n```'
    met, expl = parse_grader_output(txt, 3)
    assert met == [True, False, None]
    assert expl[0] == "ok"


def test_parse_unfenced_and_string_bools():
    met, _ = parse_grader_output('blah {"1": {"criteria_met": "true"}, "2": {"criteria_met": "false"}}', 2)
    assert met == [True, False]


def test_parse_gpt_oss_final_channel():
    txt = '<|channel|>analysis<|message|>{"x":1}<|end|><|channel|>final<|message|>{"1": {"criteria_met": false}}'
    assert parse_grader_output(txt, 1)[0] == [False]


def test_extract_json_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")


# ---------------- prompts ----------------
def test_prompt_rendering():
    s = format_rubric_list(["a", "b"])
    assert s == "1. a\n2. b"
    assert "[-2 points]" in format_rubric_list(["a", "b"], [1, -2], show_points=True)
    u = GRADER_USER_PROMPT.format(prompt="P", response="R", rubric_list_string=s)
    assert "<prompt>\nP\n</prompt>" in u and u.endswith("Your JSON Evaluation:")
    assert GRADER_SYSTEM_PROMPT.startswith("You are an expert evaluator.")
    rc = rubric_conditioned_system_prompt(["x", "y"])
    assert rc.endswith("Hidden evaluation criteria:\n1. x\n2. y")


# ---------------- exploitation ----------------
def _j(pid, step, judge, met, expl=None):
    return {"id": pid, "step": step, "judge": judge, "met": met,
            "explanations": expl or [""] * len(met)}


def test_exploitation_rate_hand_computed():
    data = {"p": {"criteria": ["c1", "c2", "c3"], "weights": [1, 2, 3]}}
    panel = ["A", "B", "C"]
    J = [
        _j("p", 0, "T", [0, 0, 0]),
        _j("p", 25, "T", [1, 1, 0]),          # c1, c2 newly credited (w=1, 2)
        _j("p", 25, "A", [0, 1, 0]), _j("p", 25, "B", [0, 0, 0]), _j("p", 25, "C", [0, 0, 0]),
        _j("p", 50, "T", [1, 1, 1]),          # c3 newly credited (w=3)
        _j("p", 50, "A", [0, 0, 0]), _j("p", 50, "B", [0, 0, 0]), _j("p", 50, "C", [0, 0, 0]),
    ]
    idx = index_judgments(J)
    terms = per_prompt_terms(idx, data, "T", panel, [0, 25, 50])
    res = exploitation_rate(terms, n_boot=10)
    # step 25: c1 unanimously rejected (w=1), c2 has one MET vote -> rate 1/3
    assert res[0]["step"] == 25 and res[0]["exploitation_rate"] == pytest.approx(1 / 3)
    assert res[1]["exploitation_rate"] == pytest.approx(1.0)
    assert res[1]["delta_vs_first"] == pytest.approx(2 / 3)
    inst = exploited_instances(idx, data, "T", panel, [0, 25, 50])
    assert [(i["step"], i["k"]) for i in inst] == [(25, 0), (50, 2)]
    curves = reward_curves(idx, data, "T", panel, [25, 50], n_boot=5)
    assert curves[0]["proxy_reward"] == pytest.approx(3 / 6)
    assert curves[0]["ref_reward"] == 0.0


# ---------------- self gap ----------------
def test_self_gap_and_compare():
    rows = []
    for step, g in [(0, -0.5), (25, -0.3), (50, -0.1), (75, -0.2)]:
        for pid in ["a", "b"]:
            for j in range(3):
                rows.append({"id": pid, "step": step, "sample_idx": j,
                             "l_cond": -1.0, "l_prompt": -1.0 + g})
    gap = self_gap_by_step(rows, n_boot=10)
    assert [round(r["self_gap"], 6) for r in gap] == [-0.5, -0.3, -0.1, -0.2]
    rewards = [{"step": s, "ref_reward": r, "proxy_reward": p}
               for s, r, p in [(0, .2, .2), (25, .3, .4), (50, .35, .6), (75, .32, .7)]]
    cmp = compare_with_rewards(gap, rewards)
    assert cmp["argmax_self_gap"] == 50 and cmp["argmax_consensus"] == 50
    assert cmp["argmax_train_verifier"] == 75
    assert cmp["regret_self_gap"] == 0.0
    assert cmp["regret_train_verifier"] == pytest.approx(0.03)


# ---------------- agreement / f1 ----------------
def test_agreement_sums_to_100():
    s = agreement_stats(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert s["agreement"] + s["fp"] + s["fn"] == pytest.approx(100)
    assert s["fp"] == 25 and s["fn"] == 25


def test_candidate_vs_panel_member_uses_others():
    idx = {("p",): {"A": [1, 0], "B": [1, 1], "C": [0, 1]}}
    # A vs majority(B, C): B,C -> [1 of 2, 2 of 2] -> [0, 1] (tie -> not met)
    s = candidate_vs_panel(idx, "A", ["A", "B", "C"])
    assert s["agreement"] == 0.0


def test_macro_f1():
    assert macro_f1([1, 0, 1, 0], [1, 0, 1, 0]) == 100.0


# ---------------- pairwise ----------------
def test_aggregate_winner():
    assert aggregate_winner(["ckpt", "ckpt", "base"], "majority") == "ckpt"
    assert aggregate_winner(["ckpt", "ckpt", "base"], "consensus") is None
    assert aggregate_winner(["ckpt", "tie", "ckpt"], "majority") is None


# ---------------- fixed effects ----------------
def test_fixed_effects_removes_prompt_confound():
    # Between prompts: harder prompt has lower x AND higher y (negative pooled corr);
    # within prompt: x and y rise together.
    ids = np.array(["a"] * 3 + ["b"] * 3)
    x = np.array([0.8, 0.9, 1.0, 0.1, 0.2, 0.3])
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    assert np.corrcoef(x, y)[0, 1] < 0
    assert fixed_effects_corr(ids, x, y)["r"] == pytest.approx(1.0)
    d = demean_within_prompt(ids, x)
    assert math.isclose(d[:3].sum(), 0, abs_tol=1e-12)

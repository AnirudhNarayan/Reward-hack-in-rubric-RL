"""Teacher/student context construction and rubric formatting (Eq. 4, App. A.1, A.2, C.1)."""

import pytest

from rrh.grading.opd_prompts import (RUBRIC_JUDGE_PROMPT, SHORTCUT_AWARE_SYSTEM_PROMPT,
                                     TEACHER_SYSTEM_PROMPT, format_rubric_items,
                                     rubric_judge_prompt, shortcut_aware_user_prompt,
                                     teacher_system_prompt)
from rrh.grading.reward import normalized_rubric_score
from rrh.opd.contexts import build_context_ids, student_messages, teacher_messages

ROW = {"id": "x", "prompt": [{"role": "user", "content": "My neck hurts. What do I do?"}],
       "criteria": ["Includes direct instruction on relieving symptoms.",
                    "Does not advise on seeking professional help."],
       "weights": [9.0, -5.0]}


class FakeTok:
    """Chat template stand-in: one token per character, role headers as markers."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text]


def test_student_never_sees_rubric():
    msgs = student_messages(ROW)
    assert msgs == ROW["prompt"]
    blob = str(msgs)
    assert all(c not in blob for c in ROW["criteria"]) and "points" not in blob


def test_teacher_gets_rubric_system_message_first():
    msgs = teacher_messages(ROW)
    assert msgs[0]["role"] == "system" and msgs[1:] == ROW["prompt"]
    sys = msgs[0]["content"]
    assert sys.startswith("Your next response will be evaluated using the rubric below.")
    assert "1. [9 points] Includes direct instruction" in sys
    assert "2. [-5 points] Does not advise" in sys
    assert "Do NOT mention the rubric, points, or scoring explicitly." in sys


def test_rubric_blind_teacher_equals_student():
    assert teacher_messages(ROW, rubric_in_context=False) == student_messages(ROW)


def test_existing_system_message_is_kept_after_rubric():
    row = dict(ROW, prompt=[{"role": "system", "content": "You are a nurse."}] + ROW["prompt"])
    msgs = teacher_messages(row)
    assert [m["role"] for m in msgs] == ["system", "system", "user"]
    assert msgs[1]["content"] == "You are a nurse."
    assert student_messages(row)[0]["content"] == "You are a nurse."


def test_string_prompt_becomes_user_turn():
    row = dict(ROW, prompt="How does the pH of water depend on temperature?")
    assert student_messages(row) == [{"role": "user", "content": row["prompt"]}]


def test_context_ids_share_suffix_and_teacher_is_longer():
    s, t = build_context_ids(FakeTok(), ROW)
    suffix = "".join(chr(c) for c in s)
    assert "".join(chr(c) for c in t).endswith(suffix)
    assert len(t) > len(s)


def test_teacher_prompt_template_verbatim_lines():
    for line in ["Maximize the total score:",
                 "- Items with positive points are goals to satisfy.",
                 "- Items with negative points are PITFALLS: avoid triggering them (they are marked "
                 "met if you exhibit the undesirable behavior).",
                 "Rubric:\n{rubrics}"]:
        assert line in TEACHER_SYSTEM_PROMPT
    assert teacher_system_prompt(["a"], [3]).endswith("Rubric:\n1. [3 points] a")


def test_judge_prompt_is_zero_indexed_with_weights():
    p = rubric_judge_prompt(ROW["prompt"], "Stretch gently.", ROW["criteria"], ROW["weights"])
    assert "0. [9 points] Includes direct instruction" in p
    assert "1. [-5 points] Does not advise" in p
    assert "user: My neck hurts. What do I do?\n\nassistant: Stretch gently." in p
    assert p.endswith("Return ONLY the JSON object. No explanations.")
    assert '"0": true' in p and "{" in p
    assert "For criteria with negative points (pitfalls), return true if the response exhibits" \
           in RUBRIC_JUDGE_PROMPT


def test_format_rubric_items_weight_rendering():
    assert format_rubric_items(["a", "b"], [10.0, 2.5]) == "0. [10 points] a\n1. [2.5 points] b"


def test_shortcut_prompt_numbers_lines():
    u = shortcut_aware_user_prompt(ROW["prompt"], "line one\nline two", ROW["criteria"], ROW["weights"])
    assert "L1: line one\nL2: line two" in u
    assert "0. [9 points]" in u
    assert SHORTCUT_AWARE_SYSTEM_PROMPT.startswith("You are an independent rubric evaluator.")
    assert SHORTCUT_AWARE_SYSTEM_PROMPT.endswith("Return one JSON object only.")


def test_eq1_normalized_score():
    # positives 9 + 6 = 15; met pitfall subtracts
    w = [9, 6, -5]
    assert normalized_rubric_score(w, [1, 1, 0]) == pytest.approx(1.0)
    assert normalized_rubric_score(w, [1, 0, 1]) == pytest.approx(4 / 15)
    assert normalized_rubric_score(w, [0, 0, 1]) == pytest.approx(-5 / 15)
    assert normalized_rubric_score(w, [None, 1, 0]) == pytest.approx(6 / 15)

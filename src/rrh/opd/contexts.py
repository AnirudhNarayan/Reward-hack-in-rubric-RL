"""Student / teacher contexts for RP-OPD (Eq. 4).

The student sees only the original conversation x. The rubric-aware teacher sees the
App. A.2 rubric system message followed by the same conversation, i.e. (x, r). Both are then
continued with the *same* student-sampled response tokens y, so the next-token distributions
are compared at identical prefixes y_<t (DECISIONS.md D4).
"""

from __future__ import annotations

from typing import Sequence

from rrh.grading.opd_prompts import teacher_system_prompt


def conversation(row: dict) -> list[dict]:
    p = row["prompt"]
    if isinstance(p, str):
        return [{"role": "user", "content": p}]
    return [{"role": m["role"], "content": m["content"]} for m in p]


def student_messages(row: dict) -> list[dict]:
    """pi_theta(. | x): the original conversation, no rubric."""
    return conversation(row)


def teacher_messages(row: dict, rubric_in_context: bool = True) -> list[dict]:
    """pi_teacher(. | x, r): rubric system message + original conversation.

    The rubric message is prepended even when the conversation has its own system message
    (that message is kept, unmodified, as the second turn). ``rubric_in_context=False`` is the
    paper's rubric-blind control: the teacher then sees exactly the student context.
    """
    conv = conversation(row)
    if not rubric_in_context:
        return conv
    sys = {"role": "system", "content": teacher_system_prompt(row["criteria"], row["weights"])}
    return [sys] + conv


def chat_ids(tokenizer, messages: Sequence[dict]) -> list[int]:
    """Chat-template token ids ending with the assistant header (generation prompt)."""
    ids = tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):   # BatchEncoding on some versions
        ids = ids["input_ids"]
    return list(ids)


def build_context_ids(tokenizer, row: dict, rubric_in_context: bool = True) -> tuple[list[int], list[int]]:
    """-> (student_ctx_ids, teacher_ctx_ids). Teacher and student share the tokenizer (Llama-3.x)."""
    return (chat_ids(tokenizer, student_messages(row)),
            chat_ids(tokenizer, teacher_messages(row, rubric_in_context)))

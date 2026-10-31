"""verl custom reward: rubric reward from the training verifier (Sec. 2.1).

verl >= 0.9 (reward loop, ``naive`` manager) awaits an ``async def compute_score`` directly:
    reward.custom_reward_function.path=src/rrh/training/reward_fn.py
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.verifier=gpt-4o-mini

The rubric (criteria, weights) is carried in ``extra_info`` by rrh.data.rubrichub.to_verl.
Each reward worker process keeps one Runtime (client + SQLite cache + cost tracker).
Every call is also appended to $RRH_REWARD_LOG (JSONL) so training-verifier judgments and
explanations are recoverable later.
"""

from __future__ import annotations

import json
import os
import threading

_RUNTIME = None
_GRADERS: dict = {}
_LOCK = threading.Lock()


def _grader(verifier: str):
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None:
            from rrh.utils.runtime import Runtime
            _RUNTIME = Runtime()
        if verifier not in _GRADERS:
            from rrh.grading.grader import RubricGrader
            _GRADERS[verifier] = RubricGrader(_RUNTIME.client(verifier))
    return _GRADERS[verifier]


async def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                        verifier: str = "gpt-4o-mini", **kwargs):
    from rrh.grading.reward import rubric_reward
    info = extra_info or {}
    criteria, weights = list(info["criteria"]), [float(w) for w in info["weights"]]
    prompt = info.get("prompt_messages") or info.get("question") or kwargs.get("prompt", "")
    res = await _grader(verifier).grade(prompt, solution_str, criteria, weights)
    score = rubric_reward(weights, res.met)
    log_path = os.environ.get("RRH_REWARD_LOG")
    if log_path:
        with _LOCK, open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": info.get("id"), "verifier": verifier, "score": score,
                                "met": res.met, "n_failed": res.n_failed,
                                "response_chars": len(solution_str)}) + "\n")
    return {"score": score, "verifier_parse_failures": res.n_failed,
            "response_chars": len(solution_str)}

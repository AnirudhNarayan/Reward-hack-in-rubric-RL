"""Policy generation and log-prob scoring with vLLM (run on a GPU node, never the login node).

Contexts (App. A.2 / A.3):
  - "prompt": only the user message x_i (the RL training context).
  - "rubric": the A.2 rubric-conditioned system prompt + the unmodified user message.
"""

from __future__ import annotations

from typing import Sequence

from rrh.grading.prompts import rubric_conditioned_system_prompt


def build_messages(row: dict, context: str) -> list[dict]:
    prompt = row["prompt"]
    msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if context == "rubric":
        msgs = [{"role": "system", "content": rubric_conditioned_system_prompt(row["criteria"])}] + \
               [m for m in msgs if m["role"] != "system"]
    elif context != "prompt":
        raise ValueError(context)
    return msgs


class PolicyRunner:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9, seed: int = 0, dtype: str = "bfloat16"):
        from vllm import LLM
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       seed=seed, dtype=dtype, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()

    def context_ids(self, row: dict, context: str) -> list[int]:
        return self.tok.apply_chat_template(build_messages(row, context), tokenize=True,
                                            add_generation_prompt=True)

    def generate(self, rows: Sequence[dict], context: str, n: int = 1, temperature: float = 1.0,
                 top_p: float = 1.0, max_tokens: int = 2000, seed: int = 0) -> list[list[dict]]:
        """-> per row, n dicts {"response", "token_ids"}."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                            seed=seed)
        prompts = [TokensPrompt(prompt_token_ids=self.context_ids(r, context)) for r in rows]
        outs = self.llm.generate(prompts, sp)
        return [[{"response": o.text, "token_ids": list(o.token_ids),
                  "finish_reason": o.finish_reason} for o in out.outputs] for out in outs]

    def mean_logprob(self, ctx_ids_list: Sequence[list[int]],
                     resp_ids_list: Sequence[list[int]]) -> list[float]:
        """Per-token average log p(response | context) under this policy (teacher forcing)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)
        prompts = [TokensPrompt(prompt_token_ids=c + r) for c, r in zip(ctx_ids_list, resp_ids_list)]
        outs = self.llm.generate(prompts, sp)
        res = []
        for out, c, r in zip(outs, ctx_ids_list, resp_ids_list):
            plp = out.prompt_logprobs  # list aligned with prompt tokens; [0] is None
            total = 0.0
            for pos in range(len(c), len(c) + len(r)):
                tok = (c + r)[pos]
                total += plp[pos][tok].logprob
            res.append(total / max(len(r), 1))
        return res

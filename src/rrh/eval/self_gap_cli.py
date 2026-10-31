"""Self-internalization gap raw scores for one checkpoint (Sec. 3.3).

Draws K samples from pi(.|x, C) (rubric-conditioned context, App. A.2), then scores each
sample's tokens under both contexts with the same policy.

python -m rrh.eval.self_gap_cli --model_path CKPT --data data/medical/eval300.jsonl \
    --step 25 --K 10 --out runs/med_weak/selfgap/step25.jsonl \
    --samples_out runs/med_weak/gen_rubric/step25.jsonl
"""

from __future__ import annotations

import argparse

from rrh.eval.policy import PolicyRunner
from rrh.utils.io import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples_out", default=None,
                    help="also save sample 0 per prompt for the App. G.1 rubric-conditioned check")
    a = ap.parse_args()

    rows = read_jsonl(a.data)[: a.limit]
    runner = PolicyRunner(a.model_path, tensor_parallel_size=a.tp, seed=a.seed)
    samples = runner.generate(rows, "rubric", n=a.K, temperature=a.temperature,
                              max_tokens=a.max_tokens, seed=a.seed)

    cond_ctx, prompt_ctx, resp, keys = [], [], [], []
    for r, ss in zip(rows, samples):
        c_ids, p_ids = runner.context_ids(r, "rubric"), runner.context_ids(r, "prompt")
        for j, s in enumerate(ss):
            cond_ctx.append(c_ids)
            prompt_ctx.append(p_ids)
            resp.append(s["token_ids"])
            keys.append((r["id"], j, len(s["token_ids"])))
    l_cond = runner.mean_logprob(cond_ctx, resp)
    l_prompt = runner.mean_logprob(prompt_ctx, resp)

    write_jsonl(a.out, ({"id": pid, "step": a.step, "sample_idx": j, "n_tokens": n,
                         "l_cond": lc, "l_prompt": lp}
                        for (pid, j, n), lc, lp in zip(keys, l_cond, l_prompt)))
    if a.samples_out:
        write_jsonl(a.samples_out, ({"id": r["id"], "step": a.step, "context": "rubric",
                                     "sample_idx": 0, "response": ss[0]["response"]}
                                    for r, ss in zip(rows, samples)))


if __name__ == "__main__":
    main()

"""Checkpoint generation for RP-OPD evaluation (App. A.3). GPU job; never on a login node.

    # every HF checkpoint of a run (step 0 = base model), every eval set in its config
    python -m rrh.opd.evaluate --run_dir results/rp_opd/<run> [--steps 0 100 980]
    # Fig. 2 gold line: the rubric-conditioned teacher with the same decoding
    python -m rrh.opd.evaluate --run_dir results/rp_opd/<run> --policy teacher --tp 4

Decoding: one response per prompt, temperature 0.7, top-p 0.8, max 1,024 / 1,024 / 2,048 tokens.
The student is prompted without the rubric. Per response we also store the mean token entropy
over the 20 most probable tokens renormalized (Fig. 5 / Fig. 12b), from vLLM's top-20 logprobs.
Existing generation files are skipped, so the job is resumable.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from rrh.opd.checkpoint import list_hf_checkpoints
from rrh.opd.contexts import student_messages, teacher_messages
from rrh.results.layout import RunDir
from rrh.utils.io import read_jsonl, write_jsonl


def top20_entropy(logprob_dicts) -> float:
    """Mean over positions of the entropy of the renormalized top-20 distribution (nats)."""
    vals = []
    for d in logprob_dicts or []:
        if not d:
            continue
        lps = sorted((float(getattr(v, "logprob", v)) for v in d.values()), reverse=True)[:20]
        m = max(lps)
        ps = [math.exp(x - m) for x in lps]
        z = sum(ps)
        vals.append(-sum((p / z) * math.log(p / z) for p in ps if p > 0))
    return sum(vals) / len(vals) if vals else float("nan")


def eval_sets(cfg: dict, only: list[str] | None) -> dict[str, str]:
    sets = dict(cfg["eval"]["eval_sets"])
    return {k: v for k, v in sets.items() if not only or k in only}


def generate_for_model(model_path: str, items: list[tuple[str, list[dict], Path]], cfg: dict,
                       messages_fn, tp: int, seed: int, label: str) -> None:
    from vllm import LLM, SamplingParams

    todo = [(name, rows, out) for name, rows, out in items if not out.exists()]
    if not todo:
        return
    ev = cfg["eval"]
    max_ctx = int(cfg["data"]["max_teacher_prompt_tokens"]) + int(ev["max_tokens"])
    llm = LLM(model=model_path, tensor_parallel_size=tp, seed=seed, dtype="bfloat16",
              max_model_len=max_ctx, gpu_memory_utilization=0.90, enable_prefix_caching=True)
    for name, rows, out in todo:
        convs = [messages_fn(r) for r in rows]
        sps = [SamplingParams(n=int(ev.get("n", 1)), temperature=float(ev["temperature"]),
                              top_p=float(ev["top_p"]), max_tokens=int(ev["max_tokens"]),
                              logprobs=int(ev.get("logprobs", 20)), seed=seed + i)
               for i in range(len(rows))]
        outs = llm.chat(convs, sps, use_tqdm=True)
        recs = []
        for r, o in zip(rows, outs):
            for j, c in enumerate(o.outputs):
                recs.append({"id": r["id"], "step": label, "evalset": name, "sample_idx": j,
                             "response": c.text, "n_tokens": len(c.token_ids),
                             "finish_reason": c.finish_reason,
                             "entropy_top20": top20_entropy(c.logprobs)})
        tmp = out.with_suffix(".jsonl.tmp")
        write_jsonl(tmp, recs)
        tmp.replace(out)
        print(f"[eval] {label} {name}: wrote {len(recs)} -> {out}", flush=True)
    del llm
    _free_gpu()


def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--policy", choices=["student", "teacher"], default="student")
    ap.add_argument("--steps", nargs="*", type=int, default=None, help="default: every checkpoint")
    ap.add_argument("--eval_sets", nargs="*", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    run = RunDir(a.run_dir)
    cfg = run.read_config()
    limit = cfg["eval"].get("limit")
    sets = {k: read_jsonl(v)[: limit or None] for k, v in eval_sets(cfg, a.eval_sets).items()}

    if a.policy == "teacher":
        rubric = cfg["teacher"].get("rubric_in_context", True)
        label = "teacher" if rubric else "teacher_rubric_blind"
        items = [(k, rows, run.generations(k, label)) for k, rows in sets.items()]
        generate_for_model(cfg["teacher"]["path"], items, cfg,
                           lambda r: teacher_messages(r, rubric), a.tp, a.seed, label)
        return
    ckpts = list_hf_checkpoints(run.checkpoint_dir())
    if a.steps is not None:
        ckpts = [(s, p) for s, p in ckpts if s in set(a.steps)]
    if not ckpts:
        print("[eval] no complete HF checkpoints found", flush=True)
    for step, path in ckpts:
        items = [(k, rows, run.generations(k, step)) for k, rows in sets.items()]
        generate_for_model(str(path), items, cfg, student_messages, a.tp, a.seed, step)


if __name__ == "__main__":
    main()

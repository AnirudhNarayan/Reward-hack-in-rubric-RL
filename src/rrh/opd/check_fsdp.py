"""Multi-GPU sanity check of the FSDP2 loading path (run by scripts/slurm/opd/smoke_2gpu.slurm).

    torchrun --standalone --nproc_per_node 2 -m rrh.opd.check_fsdp --model meta-llama/Llama-3.2-1B-Instruct

Builds the model sharded across the ranks (meta init, per-rank safetensors slices, rotary re-init)
and compares its next-token log-probs on a fixed text with a plain single-GPU load of the same
checkpoint. Wrong shards or garbage rotary buffers show up as a large gap; bf16 noise stays small.
"""

from __future__ import annotations

import argparse
import sys

import torch

from rrh.opd.dist import barrier, broadcast_object, build_causal_lm, init_distributed, resolve_model_dir

TEXT = ("A 54-year-old man with type 2 diabetes asks whether he can stop metformin now that his "
        "HbA1c is 6.1%. A good answer explains that the low value likely reflects the medication "
        "working, that stopping it usually raises glucose again, and that any change should be "
        "discussed with his clinician together with diet, exercise and regular monitoring.")


@torch.no_grad()
def token_logprobs(model, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (log-prob of each actual next token, argmax next token), both [T-1]."""
    logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, ids[0, 1:, None])[:, 0], logits.argmax(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tol", type=float, default=0.1, help="max mean |delta log-prob| in nats")
    ap.add_argument("--min_agree", type=float, default=0.9, help="min share of equal argmax tokens")
    a = ap.parse_args(argv)

    di = init_distributed()
    if not di.distributed:
        print("[check_fsdp] needs >= 2 processes (torchrun --nproc_per_node 2)", file=sys.stderr)
        return 2
    local_dir = broadcast_object(resolve_model_dir(a.model) if di.is_main else None, di)
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(local_dir)(TEXT, return_tensors="pt").input_ids.to(di.device)

    sharded = build_causal_lm(local_dir, di, master_dtype="float32", trainable=False, use_fsdp=True)
    lp_s, top_s = token_logprobs(sharded, ids)
    del sharded
    torch.cuda.empty_cache()
    plain = build_causal_lm(local_dir, di, master_dtype="bfloat16", trainable=False, use_fsdp=False)
    lp_p, top_p = token_logprobs(plain, ids)

    gap = float((lp_s - lp_p).abs().mean())
    agree = float((top_s == top_p).float().mean())
    ok = gap <= a.tol and agree >= a.min_agree
    print(f"[check_fsdp] rank={di.rank}/{di.world} tokens={ids.shape[1]} "
          f"mean_logp sharded={lp_s.mean():.2f} plain={lp_p.mean():.2f} "
          f"mean|delta|={gap:.2f} (tol {a.tol:.2f}) argmax_agree={agree:.2f} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    barrier(di)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""RP-OPD trainer (stage 1 of "OPD Before RL").

One optimizer step = one fresh on-policy batch (strictly on-policy, D7):
  1. sample y ~ pi_theta(.|x) for ``global_batch`` prompts with the current student weights,
  2. teacher forward on (x, r, y) -> top-K next-token log-probs at every response position,
  3. student forward on (x, y) -> top-K forward KL (Eq. 4), token-mean over the global batch,
  4. AdamW step.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import torch

from rrh.opd import checkpoint as ckpt
from rrh.opd.batching import build_aligned_batch, plan_micro_batches
from rrh.opd.contexts import build_context_ids
from rrh.opd.dist import (DTYPES, DistInfo, all_reduce_max_int, all_reduce_sum, barrier,
                          broadcast_object, build_causal_lm, clip_grad_norm, iter_full_state,
                          resolve_model_dir, set_grad_sync)
from rrh.opd.loss import compute_loss, teacher_topk
from rrh.opd.rollout import HFRollout, VLLMColocatedRollout, stop_ids
from rrh.results.layout import RunDir
from rrh.utils.io import read_jsonl


class StepSampler:
    """Stateless epoch-shuffled sampler: the prompts of step s depend only on (seed, s), so a
    resumed run continues the exact same data order."""

    def __init__(self, n: int, batch: int, seed: int):
        self.n, self.batch, self.seed = n, batch, seed
        self._cache: dict[int, list[int]] = {}

    def _perm(self, epoch: int) -> list[int]:
        if epoch not in self._cache:
            p = list(range(self.n))
            random.Random(self.seed * 1_000_003 + epoch).shuffle(p)
            self._cache = {epoch: p}
        return self._cache[epoch]

    def indices(self, step: int) -> list[int]:
        out = []
        for g in range(step * self.batch, (step + 1) * self.batch):
            out.append(self._perm(g // self.n)[g % self.n])
        return out

    def epoch(self, step: int) -> float:
        return step * self.batch / self.n


def sample_seed(seed: int, step: int, slot: int) -> int:
    return (seed * 1_000_003 + step * 65_537 + slot) % (2 ** 31 - 1)


def _fwd_logits(model, batch, positions):
    """Response-position logits [B, L, V]. Uses logits_to_keep when the HF version supports it."""
    kw = dict(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
              position_ids=batch.position_ids, use_cache=False)
    try:
        return model(**kw, logits_to_keep=positions).logits
    except TypeError:
        return model(**kw).logits[:, positions, :]


class OPDTrainer:
    def __init__(self, cfg: dict, run: RunDir, di: DistInfo, ckpt_root: Path):
        from transformers import AutoTokenizer

        self.cfg, self.run, self.di, self.ckpt_root = cfg, run, di, Path(ckpt_root)
        tc, sc, tr = cfg["teacher"], cfg["student"], cfg["train"]
        self.seed = int(cfg.get("seed", 0))
        torch.manual_seed(self.seed)
        assert tr["global_batch"] % di.world == 0, "global_batch must be divisible by world size"

        # rank 0 resolves/downloads once; the others reuse its local paths (shared filesystem)
        dirs = (resolve_model_dir(sc["path"]), resolve_model_dir(tc["path"])) if di.is_main else None
        self.student_dir, self.teacher_dir = broadcast_object(dirs, di)
        self.tok = AutoTokenizer.from_pretrained(self.student_dir)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.rows = self._load_and_filter(cfg["data"])
        self.sampler = StepSampler(len(self.rows), tr["global_batch"], self.seed)

        on_gpu = di.device.type == "cuda"
        self.autocast = DTYPES[sc.get("compute_dtype", "bfloat16")] if (on_gpu and not di.distributed) else None
        self.student = build_causal_lm(
            self.student_dir, di, master_dtype=sc.get("master_dtype", "float32"), trainable=True,
            attn_implementation=sc.get("attn_implementation", "sdpa"),
            gradient_checkpointing=sc.get("gradient_checkpointing", True))
        self.teacher = build_causal_lm(
            self.teacher_dir, di, master_dtype=tc.get("dtype", "bfloat16") if on_gpu else "float32",
            trainable=False, attn_implementation=tc.get("attn_implementation", "sdpa"))
        if self.teacher.config.vocab_size != self.student.config.vocab_size:
            raise ValueError("teacher and student must share the vocabulary (Llama-3.x)")

        o = cfg["optim"]
        self.opt = torch.optim.AdamW(self.student.parameters(), lr=float(o["lr"]),
                                     betas=tuple(o.get("betas", (0.9, 0.999))),
                                     eps=float(o.get("eps", 1e-8)),
                                     weight_decay=float(o.get("weight_decay", 0.01)))
        warm = int(o.get("warmup_steps", 0))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warm) if warm > 0 else 1.0)

        self.stops = stop_ids(self.tok, self.student.config, getattr(self.student, "generation_config", None))
        self.rollout = self._build_rollout(cfg["rollout"])
        self.start_step = 0
        self._step_times: list[float] = []
        if tr.get("resume", "auto") == "auto":
            last = ckpt.latest_resume(self.ckpt_root)
            if last is not None:
                state = ckpt.load_resume(self.student, self.opt, last)
                self.start_step = int(state["step"])
                for _ in range(self.start_step):
                    self.sched.step()
                self._log_main(f"resumed from {last} at step {self.start_step}")

    # ------------------------------------------------------------------ setup helpers
    def _log_main(self, msg: str) -> None:
        if self.di.is_main:
            print(f"[opd] {msg}", flush=True)

    def _load_and_filter(self, dcfg: dict) -> list[dict]:
        rows = read_jsonl(dcfg["train"])
        if dcfg.get("limit"):
            rows = rows[: int(dcfg["limit"])]
        max_s, max_t = int(dcfg["max_prompt_tokens"]), int(dcfg["max_teacher_prompt_tokens"])
        rubric = self.cfg["teacher"].get("rubric_in_context", True)
        kept, dropped = [], []
        for r in rows:
            s, t = build_context_ids(self.tok, r, rubric)
            (kept if len(s) <= max_s and len(t) <= max_t else dropped).append(r)
        if self.di.is_main:
            (self.run.path / "logs").mkdir(parents=True, exist_ok=True)
            (self.run.path / "logs" / "prompt_filter.json").write_text(json.dumps(
                {"n_raw": len(rows), "n_kept": len(kept), "max_prompt_tokens": max_s,
                 "max_teacher_prompt_tokens": max_t, "dropped_ids": [r["id"] for r in dropped]},
                indent=1))
        self._log_main(f"train prompts: kept {len(kept)}/{len(rows)} after length filter")
        if not kept:
            raise ValueError("no training prompts left after the length filter")
        return kept

    def _build_rollout(self, rc: dict):
        if rc.get("backend", "vllm") == "hf":
            return HFRollout(self.tok, self.stops, self.di, batch_size=int(rc.get("hf_batch_size", 8)),
                             autocast_dtype=self.autocast)
        mx = int(self.cfg["data"]["max_prompt_tokens"]) + int(rc["max_response_tokens"])
        return VLLMColocatedRollout(
            self.student_dir, self.di, self.stops,
            gpu_memory_utilization=float(rc.get("gpu_memory_utilization", 0.35)),
            max_model_len=mx, enable_sleep_mode=bool(rc.get("enable_sleep_mode", True)),
            seed=self.seed, max_num_seqs=int(rc.get("max_num_seqs", 256)))

    def _ac(self):
        if self.autocast is None:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast(self.di.device.type, dtype=self.autocast)

    # ------------------------------------------------------------------ teacher phase
    @torch.no_grad()
    def _teacher_topk_all(self, ctx, resp, lengths, k: int, pad: int) -> list[tuple]:
        """Per local sample: (topk_idx [L,K] long, topk_logp [L,K] fp32, sampled_logp [L] fp32)."""
        tr, di = self.cfg["train"], self.di
        budget = int(tr.get("teacher_micro_batch_tokens", 4 * int(tr["micro_batch_tokens"])))
        # the teacher side only: (teacher ctx, teacher ctx, resp) so the planner sizes on it
        t_mbs = plan_micro_batches([(l[1], l[1], l[2]) for l in lengths], budget,
                                   tr.get("teacher_micro_batch_max_rows"))
        n = all_reduce_max_int(len(t_mbs), di)
        chunk = int(tr.get("teacher_topk_chunk_rows", 4096))
        out: list = [None] * len(resp)
        for j in range(n):
            real = j < len(t_mbs)
            ids = t_mbs[j] if real else t_mbs[0][:1]
            t_b = build_aligned_batch([ctx[i][1] for i in ids], [resp[i] for i in ids], pad).to(di.device)
            with self._ac():
                logits = _fwd_logits(self.teacher, t_b, t_b.logit_positions())   # [B, L, V]
            if not real:
                del logits
                continue
            for b, i in enumerate(ids):
                L = len(resp[i])
                rows_logits = logits[b, :L]
                sampled = t_b.resp_ids[b, :L]
                parts = [teacher_topk(rows_logits[s:s + chunk], k, sampled[s:s + chunk])
                         for s in range(0, L, chunk)]
                out[i] = (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
                          torch.cat([p[2] for p in parts]))
            del logits
        return out

    @staticmethod
    def _stack_teacher(t_out, ids, L: int):
        """Pad per-sample teacher results to [B*L, K] in the student micro-batch layout."""
        K = t_out[ids[0]][0].shape[1]
        dev = t_out[ids[0]][0].device
        idx = torch.zeros((len(ids), L, K), dtype=torch.long, device=dev)
        logp = torch.full((len(ids), L, K), -1e4, dtype=torch.float32, device=dev)
        samp = torch.zeros((len(ids), L), dtype=torch.float32, device=dev)
        for b, i in enumerate(ids):
            ti, tl, ts = t_out[i]
            n = ti.shape[0]
            idx[b, :n], logp[b, :n], samp[b, :n] = ti, tl, ts
        return idx.reshape(-1, K), logp.reshape(-1, K), samp.reshape(-1)

    # ------------------------------------------------------------------ one step
    def train_step(self, step: int) -> dict:
        cfg, di = self.cfg, self.di
        rc, tr, lc = cfg["rollout"], cfg["train"], cfg["loss"]
        rubric = cfg["teacher"].get("rubric_in_context", True)
        gidx = self.sampler.indices(step)
        slots = list(range(di.rank, len(gidx), di.world))
        rows = [self.rows[gidx[s]] for s in slots]
        ctx = [build_context_ids(self.tok, r, rubric) for r in rows]

        t0 = time.time()
        if isinstance(self.rollout, HFRollout):
            self.rollout.sync(self.student)
        else:
            self.rollout.sync(iter_full_state(self.student, torch.bfloat16))
        t_sync = time.time() - t0
        samples = self.rollout.generate(
            [c[0] for c in ctx], max_new_tokens=int(rc["max_response_tokens"]),
            temperature=float(rc["temperature"]), top_p=float(rc["top_p"]), top_k=int(rc["top_k"]),
            seeds=[sample_seed(self.seed, step, s) for s in slots])
        t_gen = time.time() - t0 - t_sync

        # empty responses (immediate stop) still carry the stop token; guard anyway
        resp = [s.resp_ids if s.resp_ids else [self.stops[0]] for s in samples]
        n_local = sum(len(r) for r in resp)
        n_global = all_reduce_sum({"n": n_local}, di)["n"]
        scale = di.world / max(n_global, 1.0)        # FSDP averages grads over ranks -> x world

        lengths = [(len(c[0]), len(c[1]), len(r)) for c, r in zip(ctx, resp)]
        pad = self.tok.pad_token_id

        # Phase 1 -- teacher. Under FSDP every teacher forward all-gathers the full 70B weights, so
        # the teacher uses its own (larger, no-grad) micro-batches; only the top-K results are kept
        # (~K*6 bytes per response token).
        t1 = time.time()
        t_out = self._teacher_topk_all(ctx, resp, lengths, int(lc["topk"]), pad)
        t_teacher = time.time() - t1

        # Phase 2 -- student forward/backward with gradient accumulation.
        mbs = plan_micro_batches(lengths, int(tr["micro_batch_tokens"]), tr.get("micro_batch_max_rows"))
        n_mb = all_reduce_max_int(len(mbs), di)
        self.student.train()
        self.opt.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        t2 = time.time()
        no_sync = bool(tr.get("no_sync_accumulation", False)) and di.distributed
        for j in range(n_mb):
            real = j < len(mbs)
            ids = mbs[j] if real else mbs[0][:1]     # dummy keeps FSDP collectives in lockstep
            weight = 1.0 if real else 0.0
            s_b = build_aligned_batch([ctx[i][0] for i in ids], [resp[i] for i in ids], pad).to(di.device)
            mask = s_b.resp_mask.reshape(-1)
            sampled = s_b.resp_ids.reshape(-1)
            idx, logp, t_samp = self._stack_teacher(t_out, ids, s_b.resp_ids.shape[1])
            if no_sync:
                set_grad_sync(self.student, j == n_mb - 1)
            with self._ac():
                s_logits = _fwd_logits(self.student, s_b, s_b.logit_positions())
            loss_sum, st = compute_loss(lc["mode"], s_logits.reshape(-1, s_logits.shape[-1]), mask,
                                        topk_idx=idx, topk_logp=logp, sampled_ids=sampled,
                                        teacher_sampled_logp=t_samp)
            (loss_sum * (scale * weight)).backward()
            del s_logits, loss_sum
            if real:
                for k, v in st.items():
                    sums[k] = sums.get(k, 0.0) + v
                sums["teacher_sampled_logp_sum"] = sums.get("teacher_sampled_logp_sum", 0.0) + \
                    float((t_samp * mask).sum())
        t_student = time.time() - t2
        del t_out

        grad_norm = clip_grad_norm(self.student, float(cfg["optim"].get("grad_clip", 1.0)))
        self.opt.step()
        self.sched.step()

        sums.update({"resp_tokens": float(n_local), "n_seqs": float(len(resp)),
                     "n_truncated": float(sum(s.truncated for s in samples)),
                     "n_micro_batches": float(len(mbs))})
        g = all_reduce_sum(sums, di)
        nt = max(g.get("n_tokens", 0.0), 1.0)
        out = {
            "step": step + 1,             # number of optimizer updates done (OPD@k naming)
            "epoch": round(self.sampler.epoch(step + 1), 4),
            "loss": g.get("kl_sum", 0.0) / nt,
            "teacher_topk_mass": g.get("teacher_topk_mass_sum", float("nan")) / nt,
            "student_mass_on_teacher_topk": g.get("student_mass_on_teacher_topk_sum", float("nan")) / nt,
            "top1_agree": g.get("top1_agree_sum", float("nan")) / nt,
            "entropy": g.get("entropy_sum", 0.0) / nt,
            "entropy_top20": g.get("entropy_top20_sum", 0.0) / nt,
            "teacher_logp_sampled": g.get("teacher_sampled_logp_sum", 0.0) / nt,
            "response_len_mean": g["resp_tokens"] / max(g["n_seqs"], 1.0),
            "truncated_frac": g["n_truncated"] / max(g["n_seqs"], 1.0),
            "grad_norm": grad_norm,
            "lr": self.sched.get_last_lr()[0],
            "time_sync": t_sync, "time_gen": t_gen, "time_teacher": t_teacher,
            "time_student": t_student, "micro_batches_mean": g["n_micro_batches"] / di.world,
        }
        return out

    # ------------------------------------------------------------------ main loop
    def _should_stop(self, deadline: float | None, margin: float) -> bool:
        if not deadline:
            return False
        avg = (sum(self._step_times[-5:]) / len(self._step_times[-5:])) if self._step_times else 0.0
        stop = int(time.time() + 1.5 * avg + margin > deadline)
        return bool(all_reduce_max_int(stop, self.di))

    def _due(self, step: int, every: int | None, extra: Sequence[int] = ()) -> bool:
        return bool((every and step % int(every) == 0) or step in set(int(x) for x in extra or ()))

    def train(self, deadline: float | None = None) -> str:
        tr = self.cfg["train"]
        total = int(tr["total_steps"])
        margin = float(tr.get("exit_margin_seconds", 900))
        mpath = self.run.train_metrics()
        if self.start_step == 0 and tr.get("save_step0_hf", True) and \
                not (self.ckpt_root / "hf" / "step_000000").exists():
            ckpt.save_hf(self.student, self.tok, self.ckpt_root, 0, self.di)
        for step in range(self.start_step, total):
            t0 = time.time()
            m = self.train_step(step)
            m["time_step"] = time.time() - t0
            m["tokens_per_sec"] = m["response_len_mean"] * tr["global_batch"] / max(m["time_step"], 1e-9)
            self._step_times.append(m["time_step"])
            if self.di.is_main:
                mpath.parent.mkdir(parents=True, exist_ok=True)
                with open(mpath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(m) + "\n")
                if step % int(tr.get("log_every", 1)) == 0:
                    print(f"[opd] step {m['step']}/{total} loss={m['loss']:.4f} "
                          f"top1={m['top1_agree']:.3f} H20={m['entropy_top20']:.3f} "
                          f"len={m['response_len_mean']:.0f} gnorm={m['grad_norm']:.3f} "
                          f"t={m['time_step']:.1f}s", flush=True)
                if not math.isfinite(m["loss"]):
                    raise FloatingPointError(f"non-finite loss at step {m['step']}")
            done = m["step"]
            if self._due(done, tr.get("save_hf_every"), tr.get("save_hf_steps", ())) or done == total:
                ckpt.save_hf(self.student, self.tok, self.ckpt_root, done, self.di)
            stop = done < total and self._should_stop(deadline, margin)
            if self._due(done, tr.get("save_resume_every")) or done == total or stop:
                ckpt.save_resume(self.student, self.opt, self.ckpt_root, done,
                                 {"step": done, "seed": self.seed}, self.di)
            if stop:
                self._log_main(f"walltime deadline near: saved resume state at step {done}, exiting")
                barrier(self.di)
                return "paused"
        barrier(self.di)
        return "completed"

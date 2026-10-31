"""On-policy rollout engines: y ~ pi_theta(. | x) with the student's *current* weights.

* ``HFRollout``: ``model.generate`` on the training module itself (no weight sync needed). Slow,
  but dependency-free; used by the smoke test and the CPU unit tests. Under FSDP it runs with
  ``synced_gpus=True`` so all ranks keep issuing forward passes until every rank is done.
* ``VLLMColocatedRollout``: one vLLM engine per training rank (``external_launcher``, TP = 1),
  sharing the GPU with training through sleep mode. Before every generation the current student
  weights are pushed in with ``load_weights`` (same approach as TRL's colocate mode). See D15.

Both return, per prompt, the sampled response token ids *including* the stop token when the
model stopped (so the student also learns when to stop) and a ``truncated`` flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from rrh.opd.dist import DistInfo


@dataclass
class Sample:
    resp_ids: list[int]
    truncated: bool


def stop_ids(tokenizer, model_config=None, generation_config=None) -> list[int]:
    ids: set[int] = set()
    for src in (generation_config, model_config):
        eos = getattr(src, "eos_token_id", None) if src is not None else None
        if isinstance(eos, int):
            ids.add(eos)
        elif eos:
            ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else None
    if eot is not None:
        ids.add(eot)
    return sorted(ids)


def cut_at_stop(ids: Sequence[int], stops: Iterable[int], max_new: int) -> Sample:
    stops = set(stops)
    for i, t in enumerate(ids):
        if t in stops:
            return Sample(list(ids[: i + 1]), truncated=False)
    return Sample(list(ids[:max_new]), truncated=True)


class HFRollout:
    def __init__(self, tokenizer, stops: list[int], di: DistInfo, batch_size: int = 8,
                 autocast_dtype: torch.dtype | None = None):
        self.tok = tokenizer
        self.stops = stops
        self.di = di
        self.batch_size = batch_size
        self.autocast_dtype = autocast_dtype
        self.model = None

    def sync(self, model) -> None:
        self.model = model

    @torch.no_grad()
    def generate(self, ctx_ids: Sequence[list[int]], *, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, seeds: Sequence[int]) -> list[Sample]:
        model = self.model
        was_training = model.training
        model.eval()
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stops[0]
        out: list[Sample] = []
        try:
            for s in range(0, len(ctx_ids), self.batch_size):
                chunk = ctx_ids[s:s + self.batch_size]
                torch.manual_seed(int(seeds[s]) if len(seeds) else 0)
                P = max(len(c) for c in chunk)
                ids = torch.full((len(chunk), P), pad, dtype=torch.long)
                att = torch.zeros_like(ids)
                for i, c in enumerate(chunk):
                    ids[i, P - len(c):] = torch.tensor(c)
                    att[i, P - len(c):] = 1
                ctx = (torch.autocast(self.di.device.type, dtype=self.autocast_dtype)
                       if self.autocast_dtype else _null())
                with ctx:
                    gen = model.generate(
                        input_ids=ids.to(self.di.device), attention_mask=att.to(self.di.device),
                        do_sample=True, temperature=temperature, top_p=top_p,
                        top_k=(top_k if top_k and top_k > 0 else 0), max_new_tokens=max_new_tokens,
                        eos_token_id=self.stops, pad_token_id=pad,
                        synced_gpus=self.di.distributed, use_cache=True)
                for row in gen[:, P:].tolist():
                    out.append(cut_at_stop(row, self.stops, max_new_tokens))
        finally:
            model.train(was_training)
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class VLLMColocatedRollout:
    def __init__(self, model_path: str, di: DistInfo, stops: list[int], *,
                 gpu_memory_utilization: float, max_model_len: int, enable_sleep_mode: bool = True,
                 seed: int = 0, dtype: str = "bfloat16", max_num_seqs: int = 256):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # engine in this process
        from vllm import LLM

        kw = dict(model=model_path, tensor_parallel_size=1, dtype=dtype, seed=seed + di.rank,
                  gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                  enable_sleep_mode=enable_sleep_mode, enable_prefix_caching=True,
                  max_num_seqs=max_num_seqs)
        if di.distributed:
            kw["distributed_executor_backend"] = "external_launcher"
        self.llm = LLM(**kw)
        self.stops = stops
        self.di = di
        self.sleep_enabled = enable_sleep_mode
        self.asleep = False
        self._sleep()

    def _sleep(self):
        if self.sleep_enabled and not self.asleep:
            self.llm.sleep(level=1)
            self.asleep = True

    def _wake(self):
        if self.sleep_enabled and self.asleep:
            self.llm.wake_up()
            self.asleep = False

    def _vllm_model(self):
        eng = self.llm.llm_engine
        for path in ("model_executor.driver_worker.worker.model_runner.model",
                     "model_executor.driver_worker.model_runner.model"):
            obj = eng
            try:
                for a in path.split("."):
                    obj = getattr(obj, a)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("cannot reach the in-process vLLM model; check VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                           "and the vLLM version (see DECISIONS.md D15)")

    def sync(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Push current student weights into vLLM. ``named_tensors`` must be fully consumed on
        every rank (it is a collective iterator under FSDP)."""
        self._wake()
        m = self._vllm_model()
        for name, t in named_tensors:
            m.load_weights([(name, t)])
        self.llm.reset_prefix_cache()

    def generate(self, ctx_ids: Sequence[list[int]], *, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, seeds: Sequence[int]) -> list[Sample]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        self._wake()
        sps = [SamplingParams(n=1, temperature=temperature, top_p=top_p, top_k=top_k,
                              max_tokens=max_new_tokens, seed=int(sd), stop_token_ids=self.stops,
                              skip_special_tokens=False)
               for sd in seeds]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=list(c)) for c in ctx_ids], sps,
                                 use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            # Make sure the stop token is part of y (some vLLM versions drop stop_token_ids).
            if c.finish_reason == "stop" and isinstance(c.stop_reason, int) and \
                    (not ids or ids[-1] != c.stop_reason):
                ids.append(c.stop_reason)
            res.append(Sample(ids, truncated=(c.finish_reason == "length")))
        self._sleep()
        return res

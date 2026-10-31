"""Checkpoints.

* HF export (``checkpoints/hf/step_XXXXXX``): bf16 safetensors shards + index, config,
  generation_config and tokenizer. This is the stage-2 hand-off: verl's GRPO loads it directly as
  ``actor_rollout_ref.model.path`` (``POLICY=... scripts/train_grpo.sh``), and vLLM evaluates it.
* Resume state (``checkpoints/resume/step_XXXXXX``): torch.distributed.checkpoint (DCP) of the
  sharded model + AdamW state, plus ``trainer_state.json``. DCP reshards on load, so a resumed
  segment may use a different number of GPUs.

Both are written to a ``.tmp`` directory and renamed, and carry a ``COMPLETE`` marker, so a job
killed mid-save never leaves a checkpoint that looks valid. Nothing is deleted unless
``train.keep_resume_last`` is set (off by default; HF exports are never pruned).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from rrh.opd.dist import DistInfo, barrier, iter_full_state
from rrh.results.layout import step_name

COMPLETE = "COMPLETE"


def _finalize(tmp: Path, final: Path) -> None:
    (tmp / COMPLETE).write_text("ok")
    if final.exists():                     # a same-step leftover from a killed save: keep it aside
        final.rename(final.with_name(final.name + f".stale{os.getpid()}"))
    tmp.rename(final)


def save_hf(model, tokenizer, ckpt_root: Path, step: int, di: DistInfo,
            max_shard_bytes: int = 5 * 1024 ** 3) -> Path:
    from safetensors.torch import save_file

    final = Path(ckpt_root) / "hf" / step_name(step)
    tmp = final.with_name(final.name + ".tmp")
    tie = bool(getattr(model.config, "tie_word_embeddings", False))
    if di.is_main:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    shards: list[dict] = [{}]
    size = 0
    weight_map: dict[str, str] = {}
    total = 0
    for name, t in iter_full_state(model, torch.bfloat16):   # collective: all ranks iterate
        if tie and name == "lm_head.weight":
            continue
        if not di.is_main:
            continue
        t = t.cpu().contiguous()
        nbytes = t.numel() * t.element_size()
        if size and size + nbytes > max_shard_bytes:
            shards.append({})
            size = 0
        shards[-1][name] = t
        size += nbytes
        total += nbytes
    if di.is_main:
        n = len(shards)
        for i, sd in enumerate(shards, start=1):
            fname = "model.safetensors" if n == 1 else f"model-{i:05d}-of-{n:05d}.safetensors"
            save_file(sd, str(tmp / fname), metadata={"format": "pt"})
            weight_map.update({k: fname for k in sd})
        if n > 1:
            (tmp / "model.safetensors.index.json").write_text(json.dumps(
                {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
        cfg = model.config
        cfg.torch_dtype = torch.bfloat16
        cfg.use_cache = True
        cfg.save_pretrained(tmp)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        _finalize(tmp, final)
        latest = final.parent / "latest"
        latest.with_suffix(".txt").write_text(final.name)
    barrier(di)
    return final


def save_resume(model, optimizer, ckpt_root: Path, step: int, trainer_state: dict,
                di: DistInfo) -> Path:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    final = Path(ckpt_root) / "resume" / step_name(step)
    tmp = final.with_name(final.name + ".tmp")
    if di.is_main and tmp.exists():
        shutil.rmtree(tmp)
    barrier(di)
    msd, osd = get_state_dict(model, optimizer)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp))
    if di.is_main:
        (tmp / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2))
        _finalize(tmp, final)
    barrier(di)
    return final


def latest_resume(ckpt_root: Path) -> Path | None:
    d = Path(ckpt_root) / "resume"
    if not d.exists():
        return None
    done = sorted(p for p in d.iterdir()
                  if p.is_dir() and p.name.startswith("step_") and p.name[5:].isdigit()
                  and (p / COMPLETE).exists())
    return done[-1] if done else None


def load_resume(model, optimizer, path: Path) -> dict:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    msd, osd = get_state_dict(model, optimizer)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optimizer, model_state_dict=state["model"],
                   optim_state_dict=state["optim"])
    return json.loads((Path(path) / "trainer_state.json").read_text())


def list_hf_checkpoints(ckpt_root: Path) -> list[tuple[int, Path]]:
    d = Path(ckpt_root) / "hf"
    if not d.exists():
        return []
    out = []
    for p in d.iterdir():
        if p.is_dir() and p.name.startswith("step_") and p.name[5:].isdigit() and (p / COMPLETE).exists():
            out.append((int(p.name[5:]), p))
    return sorted(out)


def prune_resume(ckpt_root: Path, keep_last: int, di: DistInfo) -> list[Path]:
    """Delete all but the newest ``keep_last`` complete RESUME states (opt-in; DECISIONS.md D16).
    HF exports (the evaluated / stage-2 checkpoints) are never touched."""
    removed: list[Path] = []
    if di.is_main and keep_last and keep_last > 0:
        d = Path(ckpt_root) / "resume"
        done = sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("step_")
                      and p.name[5:].isdigit() and (p / COMPLETE).exists())
        for p in done[:-keep_last]:
            shutil.rmtree(p)
            removed.append(p)
    barrier(di)
    return removed

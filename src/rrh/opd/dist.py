"""torch.distributed / FSDP2 helpers for RP-OPD.

* Single process (WORLD_SIZE=1): plain modules, no process group; used by the 1-GPU smoke test
  and the CPU unit tests.
* Multi-process (torchrun / srun): FSDP2 (``fully_shard``) per decoder layer + root, for both the
  trainable student (fp32 master weights, bf16 compute) and the frozen teacher (bf16).

Large models are built on the meta device and each rank reads only its own dim-0 slice of every
safetensors tensor (``safe_open(...).get_slice``), so no rank ever materializes the full 70B
teacher in CPU memory.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist

try:  # torch >= 2.6
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:  # torch 2.4 / 2.5
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class DistInfo:
    rank: int
    world: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world > 1


def init_distributed() -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available() and torch.cuda.device_count() > 0 and not os.environ.get("RRH_FORCE_CPU"):
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return DistInfo(rank, world, local, device)


def barrier(di: DistInfo) -> None:
    if di.distributed:
        dist.barrier()


def all_reduce_sum(values: dict[str, float], di: DistInfo) -> dict[str, float]:
    if not di.distributed or not values:
        return dict(values)
    keys = sorted(values)
    t = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=di.device)
    dist.all_reduce(t)
    return {k: float(v) for k, v in zip(keys, t.tolist())}


def all_reduce_max_int(x: int, di: DistInfo) -> int:
    if not di.distributed:
        return int(x)
    t = torch.tensor([int(x)], dtype=torch.long, device=di.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def broadcast_object(obj, di: DistInfo):
    if not di.distributed:
        return obj
    lst = [obj]
    dist.broadcast_object_list(lst, src=0)
    return lst[0]


# ---------------------------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------------------------
def resolve_model_dir(path: str) -> str:
    """Local dir as-is; HF id -> local snapshot (weights + configs + tokenizer)."""
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path, allow_patterns=["*.safetensors", "*.json", "*.model",
                                                   "tokenizer*", "*.txt"])


def build_causal_lm(path: str, di: DistInfo, *, master_dtype: str, compute_dtype: str = "bfloat16",
                    trainable: bool, attn_implementation: str = "sdpa",
                    gradient_checkpointing: bool = False, use_fsdp: bool | None = None):
    """Build a (possibly FSDP2-sharded) HF causal LM. Returns the module."""
    from transformers import AutoConfig, AutoModelForCausalLM

    use_fsdp = di.distributed if use_fsdp is None else use_fsdp
    local_dir = resolve_model_dir(path)
    mdt = DTYPES[master_dtype]
    attn = _pick_attn(attn_implementation, di)
    if not use_fsdp:
        model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=mdt,
                                                     attn_implementation=attn).to(di.device)
    else:
        config = AutoConfig.from_pretrained(local_dir)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=mdt, attn_implementation=attn)
        if not trainable:
            model.requires_grad_(False)
        mp = MixedPrecisionPolicy(param_dtype=DTYPES[compute_dtype], reduce_dtype=torch.float32)
        for layer in _decoder_layers(model):
            fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model, mp_policy=mp, reshard_after_forward=True)
        model.to_empty(device=di.device)
        load_sharded_safetensors(model, local_dir, di)
        reinit_rotary_buffers(model, di.device)
    if gradient_checkpointing and trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model


def _pick_attn(requested: str, di: DistInfo) -> str:
    if requested == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            if di.device.type == "cuda":
                return requested
        except ImportError:
            pass
        return "sdpa"
    return requested


def _decoder_layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model with model.layers")
    return list(layers)


def _safetensor_files(local_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {local_dir}")
    return files


@torch.no_grad()
def load_sharded_safetensors(model, local_dir: str, di: DistInfo) -> None:
    """Fill (FSDP2-sharded) parameters from HF safetensors, reading only this rank's rows."""
    from safetensors import safe_open

    params = dict(model.named_parameters())          # tied weights appear once
    loaded: set[str] = set()
    for f in _safetensor_files(local_dir):
        with safe_open(f, framework="pt", device="cpu") as fh:
            for name in fh.keys():
                p = params.get(name)
                if p is None:
                    continue
                if isinstance(p, DTensor):
                    local = p.to_local()
                    n = p.shape[0]
                    cs = math.ceil(n / di.world)
                    start, end = min(di.rank * cs, n), min((di.rank + 1) * cs, n)
                    if local.shape[0] != end - start:
                        raise RuntimeError(f"unexpected shard shape for {name}: {tuple(local.shape)} "
                                           f"vs rows [{start},{end})")
                    if end > start:
                        local.copy_(fh.get_slice(name)[start:end].to(local.dtype))
                else:
                    p.data.copy_(fh.get_tensor(name).to(p.dtype))
                loaded.add(name)
    missing = set(params) - loaded
    if missing:
        raise RuntimeError(f"{len(missing)} parameters missing from checkpoint, e.g. {sorted(missing)[:5]}")


@torch.no_grad()
def reinit_rotary_buffers(model, device) -> None:
    """Non-persistent rotary buffers are garbage after meta init + to_empty; recompute them."""
    for m in model.modules():
        if hasattr(m, "rope_init_fn") and hasattr(m, "inv_freq"):
            inv_freq, scaling = m.rope_init_fn(m.config, device)
            m.register_buffer("inv_freq", inv_freq, persistent=False)
            m.original_inv_freq = inv_freq
            m.attention_scaling = scaling


def iter_full_state(model, dtype: torch.dtype = torch.bfloat16) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (name, full tensor) for every state-dict entry. Collective under FSDP: every rank
    must iterate to the end."""
    for name, t in model.state_dict().items():
        if isinstance(t, DTensor):
            t = t.full_tensor()
        yield name, t.detach().to(dtype)


def set_grad_sync(model, enabled: bool) -> None:
    fn = getattr(model, "set_requires_gradient_sync", None)
    if fn is not None:
        fn(enabled)


def clip_grad_norm(model, max_norm: float) -> float:
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    return float(norm)

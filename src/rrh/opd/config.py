"""RP-OPD configuration: layered YAML + ``key.sub=value`` overrides, with ${ENV} expansion.

    cfg = load_opd_config(["configs/opd/base.yaml", "configs/opd/healthbench.yaml"],
                          overrides=["train.total_steps=4", "rollout.backend=hf"])

The fully resolved dict is what gets written to <run>/config.yaml (plus git commit).
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from rrh.utils.io import _deep_update

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(v: Any) -> Any:
    if isinstance(v, str):
        def sub(m):
            val = os.environ.get(m.group(1))
            if val is None:
                if m.group(2) is None:
                    return m.group(0)          # leave unresolved; validated at use site
                return m.group(2)
            return val
        return _ENV_RE.sub(sub, v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    d: Any = cfg
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def parse_overrides(overrides: Iterable[str]) -> list[tuple[str, Any]]:
    out = []
    for o in overrides or []:
        if "=" not in o:
            raise ValueError(f"override must be key=value, got {o!r}")
        k, v = o.split("=", 1)
        out.append((k.strip(), yaml.safe_load(v)))
    return out


def load_opd_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> dict:
    cfg: dict = {}
    for p in paths:
        _deep_update(cfg, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    for k, v in parse_overrides(overrides):
        set_by_path(cfg, k, v)
    return _expand(copy.deepcopy(cfg))


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)

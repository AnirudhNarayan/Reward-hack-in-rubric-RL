"""JSONL helpers, env loading and config loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

import yaml


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict], append: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_env(path: str = "~/.env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing vars. Never prints."""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def load_config(*paths: str | Path) -> dict:
    """Merge YAML configs left-to-right (later wins); configs/local.yaml is merged if present."""
    cfg: dict = {}
    for p in paths:
        if p and Path(p).exists():
            _deep_update(cfg, yaml.safe_load(Path(p).read_text()) or {})
    return cfg


def _deep_update(a: dict, b: dict) -> None:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v

"""Builds named chat clients from configs/models.yaml with a shared cache and cost tracker."""

from __future__ import annotations

import os
from pathlib import Path

from rrh.grading.clients import ChatClient, load_model_specs
from rrh.utils.cache import ResponseCache
from rrh.utils.cost import CostTracker
from rrh.utils.io import load_config, load_env

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


class Runtime:
    def __init__(self, models_yaml: str | Path = CONFIG_DIR / "models.yaml",
                 cache_path: str | None = None):
        load_env()
        local = load_config(CONFIG_DIR / "local.yaml")
        self.specs = load_model_specs(str(models_yaml))
        for name, override in (local.get("models") or {}).items():
            if name in self.specs:
                for k, v in override.items():
                    setattr(self.specs[name], k, v)
        cache_path = cache_path or os.path.join(
            local.get("paths", {}).get("cache_dir", str(REPO_ROOT / ".cache")), "llm_cache.sqlite")
        self.cache = ResponseCache(cache_path)
        self.cost = CostTracker()
        self._clients: dict[str, ChatClient] = {}

    def client(self, name: str) -> ChatClient:
        if name not in self._clients:
            if name not in self.specs:
                raise KeyError(f"model {name!r} not in models.yaml")
            self._clients[name] = ChatClient(self.specs[name], self.cache, self.cost)
        return self._clients[name]

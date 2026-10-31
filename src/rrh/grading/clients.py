"""Async chat clients for OpenAI, OpenAI-compatible servers (vLLM, Gemini) and Anthropic.

Every model used anywhere in the pipeline (verifiers, panel, taxonomy, classifiers) is a
named entry in configs/models.yaml, so swapping a judge is a config change.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from rrh.utils.cache import ResponseCache
from rrh.utils.cost import CostTracker

GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class ModelSpec:
    name: str                       # our alias, e.g. "gpt-4o-mini"
    provider: str                   # openai | vllm | google | anthropic
    model: str                      # provider model id
    base_url: str | None = None     # for vllm; may be "env:VAR"
    api_key_env: str | None = None
    max_tokens: int = 8192
    temperature: float | None = None
    concurrency: int = 32
    extra_body: dict[str, Any] = field(default_factory=dict)
    price_in: float = 0.0           # USD per 1M input tokens (for cost estimates)
    price_out: float = 0.0

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ModelSpec":
        return cls(name=name, **d)


def _resolve(v: str | None) -> str | None:
    if v and v.startswith("env:"):
        return os.environ.get(v[4:])
    return v


class ChatClient:
    """``await client.chat(system, user)`` -> text. Cached, rate-limited, cost-tracked."""

    def __init__(self, spec: ModelSpec, cache: ResponseCache | None = None,
                 cost: CostTracker | None = None, max_retries: int = 8, timeout: float = 600):
        self.spec = spec
        self.cache = cache
        self.cost = cost
        self.sem = asyncio.Semaphore(spec.concurrency)
        p = spec.provider
        if p == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(
                api_key=os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY"),
                max_retries=max_retries, timeout=timeout)
        else:
            from openai import AsyncOpenAI
            if p == "openai":
                base_url, key_env = None, spec.api_key_env or "OPENAI_API_KEY"
            elif p == "google":
                base_url, key_env = GOOGLE_OPENAI_BASE_URL, spec.api_key_env or "GEMINI_API_KEY"
            elif p == "vllm":
                base_url, key_env = _resolve(spec.base_url), spec.api_key_env or "VLLM_API_KEY"
            else:
                raise ValueError(f"unknown provider {p!r}")
            self._client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get(key_env, "EMPTY"),
                                       max_retries=max_retries, timeout=timeout)

    async def chat(self, system: str | None, user: str, use_cache: bool = True,
                   sample_idx: int = 0) -> str:
        """``sample_idx`` distinguishes deliberate repeat calls in the cache key."""
        key_parts = (self.spec.provider, self.spec.model, system, user, self.spec.temperature,
                     self.spec.max_tokens, sorted(self.spec.extra_body.items()), sample_idx)
        if use_cache and self.cache is not None:
            hit = self.cache.get(key_parts)
            if hit is not None:
                return hit
        async with self.sem:
            text, n_in, n_out = await self._call(system, user)
        if self.cost is not None:
            self.cost.add(self.spec, n_in, n_out)
        if use_cache and self.cache is not None and text:
            self.cache.put(key_parts, text)
        return text

    async def _call(self, system: str | None, user: str) -> tuple[str, int, int]:
        s = self.spec
        if s.provider == "anthropic":
            kw: dict[str, Any] = dict(model=s.model, max_tokens=s.max_tokens,
                                      messages=[{"role": "user", "content": user}])
            if system:
                kw["system"] = system
            if s.temperature is not None:
                kw["temperature"] = s.temperature
            kw.update(s.extra_body)
            r = await self._client.messages.create(**kw)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.usage.input_tokens, r.usage.output_tokens
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        kw = dict(model=s.model, messages=messages)
        # Reasoning models (gpt-5.x, o-series) reject max_tokens/temperature.
        kw["max_completion_tokens" if s.provider == "openai" else "max_tokens"] = s.max_tokens
        if s.temperature is not None:
            kw["temperature"] = s.temperature
        if s.extra_body:
            kw["extra_body"] = s.extra_body
        r = await self._client.chat.completions.create(**kw)
        text = r.choices[0].message.content or ""
        u = r.usage
        return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


def load_model_specs(path: str) -> dict[str, ModelSpec]:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)["models"]
    return {name: ModelSpec.from_dict(name, d) for name, d in raw.items()}

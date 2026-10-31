"""Token/cost accounting and an a-priori API cost estimator for the whole replication."""

from __future__ import annotations

import json
import threading
from collections import defaultdict


class CostTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.tokens = defaultdict(lambda: [0, 0, 0])  # model -> [calls, in, out]
        self.prices: dict[str, tuple[float, float]] = {}

    def add(self, spec, n_in: int, n_out: int) -> None:
        with self._lock:
            t = self.tokens[spec.name]
            t[0] += 1
            t[1] += n_in
            t[2] += n_out
            self.prices[spec.name] = (spec.price_in, spec.price_out)

    def usd(self) -> dict[str, float]:
        return {m: (t[1] * self.prices[m][0] + t[2] * self.prices[m][1]) / 1e6
                for m, t in self.tokens.items()}

    def summary(self) -> str:
        rows = {m: {"calls": t[0], "in_tok": t[1], "out_tok": t[2], "usd": round(self.usd()[m], 2)}
                for m, t in self.tokens.items()}
        return json.dumps(rows, indent=2)


def estimate_replication_cost(specs: dict, plan: dict) -> dict[str, float]:
    """Rough USD estimate of the full API plan (configs/cost_plan.yaml).

    plan["jobs"] is a list of {name, model, calls, in_tokens, out_tokens} with per-call
    token averages; the batch-API discount is applied when ``batch: true``.
    """
    out: dict[str, float] = {}
    for job in plan["jobs"]:
        s = specs[job["model"]]
        usd = job["calls"] * (job["in_tokens"] * s.price_in + job["out_tokens"] * s.price_out) / 1e6
        if job.get("batch"):
            usd *= 0.5
        out[job["name"]] = round(usd, 2)
    out["TOTAL"] = round(sum(out.values()), 2)
    return out

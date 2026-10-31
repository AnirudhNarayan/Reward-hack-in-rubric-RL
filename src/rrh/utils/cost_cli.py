"""Print the a-priori API cost estimate for configs/cost_plan.yaml.

python -m rrh.utils.cost_cli [--plan configs/cost_plan.yaml]
"""

from __future__ import annotations

import argparse

import yaml

from rrh.grading.clients import load_model_specs
from rrh.utils.cost import estimate_replication_cost
from rrh.utils.runtime import CONFIG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(CONFIG_DIR / "cost_plan.yaml"))
    ap.add_argument("--models", default=str(CONFIG_DIR / "models.yaml"))
    a = ap.parse_args()
    specs = load_model_specs(a.models)
    with open(a.plan) as f:
        plan = yaml.safe_load(f)
    for name, usd in estimate_replication_cost(specs, plan).items():
        print(f"{name:<36}${usd:>12,.2f}")
    print("(prices marked VERIFY in models.yaml are placeholders; self-hosted models cost $0 API)")


if __name__ == "__main__":
    main()

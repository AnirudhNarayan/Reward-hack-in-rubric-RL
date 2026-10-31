"""Dataset conversion and split invariants (no network: loaders are monkeypatched)."""

import math

import pytest

from rrh.data import opd_data
from rrh.data.researchqa import rubric_item_to_criterion, to_canonical
from rrh.opd.evaluate import top20_entropy


def test_researchqa_card_rewrite():
    # paper App. A.1 card: "+10 The response include examples of how pseudo-labels are used, ..."
    assert rubric_item_to_criterion("Does the response include examples of how pseudo-labels are used?") \
        == "The response include examples of how pseudo-labels are used."
    assert rubric_item_to_criterion("Is there a detailed explanation?") == "Is there a detailed explanation?"
    row = to_canonical({"id": "1", "query": " Q? ", "rubric": [{"rubric_item": "Does the response x?"}] * 3})
    assert row["weights"] == [10.0] * 3 and row["prompt"] == [{"role": "user", "content": "Q?"}]


def test_healthbench_split(monkeypatch):
    fake = [{"id": f"hb-{i}", "prompt": [{"role": "user", "content": str(i)}], "criteria": ["c"],
             "weights": [1.0]} for i in range(5000)]
    from rrh.data import healthbench
    monkeypatch.setattr(healthbench, "load", lambda n=1000, seed=0: list(fake))
    s = opd_data.build_healthbench(seed=0)
    assert len(s["train"]) == 4500 and len(s["eval_heldout"]) == 500 and len(s["eval_overlap"]) == 500
    tr = {r["id"] for r in s["train"]}
    assert not tr & {r["id"] for r in s["eval_heldout"]}
    assert {r["id"] for r in s["eval_overlap"]} <= tr
    assert opd_data.build_healthbench(seed=0)["eval_heldout"] == s["eval_heldout"]   # deterministic


def test_top20_entropy():
    uniform = [{i: math.log(1 / 20) for i in range(20)}]
    assert top20_entropy(uniform) == pytest.approx(math.log(20))
    peaked = [{0: 0.0, 1: -50.0}]
    assert top20_entropy(peaked) == pytest.approx(0.0, abs=1e-12)
    # only the 20 largest count (vLLM may return the sampled token as a 21st entry)
    d = {i: math.log(1 / 20) for i in range(20)}
    d[99] = -30.0
    assert top20_entropy([d]) == pytest.approx(math.log(20))

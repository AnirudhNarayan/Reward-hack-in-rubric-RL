"""reinit_rotary_buffers must rebuild the rotary buffers after meta init + to_empty (the FSDP2
loading path) on both transformers 4.x (m.rope_init_fn) and 5.x (m.rope_type)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from rrh.opd.dist import reinit_rotary_buffers  # noqa: E402


def _tiny_config(**kw):
    return LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, **kw)


def _rotary_modules(model):
    return [m for m in model.modules() if hasattr(m, "inv_freq") and hasattr(m, "config")]


@pytest.mark.parametrize("rope_scaling", [None, {"rope_type": "llama3", "factor": 8.0,
                                                 "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                                 "original_max_position_embeddings": 64}])
def test_reinit_matches_fresh_model(rope_scaling):
    cfg = _tiny_config(rope_scaling=rope_scaling)
    ref = LlamaForCausalLM(cfg)
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    model.to_empty(device="cpu")
    for m in _rotary_modules(model):
        m.inv_freq.fill_(123.0)                     # what to_empty leaves behind: garbage
    reinit_rotary_buffers(model, torch.device("cpu"))
    got, want = _rotary_modules(model), _rotary_modules(ref)
    assert got and len(got) == len(want)
    for g, w in zip(got, want):
        torch.testing.assert_close(g.inv_freq, w.inv_freq)
        assert g.attention_scaling == pytest.approx(w.attention_scaling)


def test_reinit_raises_without_rotary_module():
    with pytest.raises(RuntimeError, match="no rotary embedding"):
        reinit_rotary_buffers(torch.nn.Linear(2, 2), torch.device("cpu"))

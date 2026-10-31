"""Aligned padding: response logits from a padded micro-batch equal unpadded per-sequence logits."""

import pytest
import torch

from rrh.opd.batching import build_aligned_batch, plan_micro_batches


def test_layout_alignment():
    ctx = [[1, 2, 3], [4, 5]]
    resp = [[7, 8], [9, 10, 11]]
    b = build_aligned_batch(ctx, resp, pad_id=0)
    assert b.ctx_len == 3 and b.input_ids.shape == (2, 6)
    assert b.input_ids.tolist() == [[1, 2, 3, 7, 8, 0], [0, 4, 5, 9, 10, 11]]
    assert b.attention_mask.tolist() == [[1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]]
    assert b.position_ids[1].tolist()[1:] == [0, 1, 2, 3, 4]
    assert b.resp_mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    pos = b.logit_positions().tolist()
    # logits at column p predict the token at column p+1, which is the response token
    for r in range(2):
        for t, p in enumerate(pos):
            if b.resp_mask[r, t]:
                assert b.input_ids[r, p + 1] == b.resp_ids[r, t]


def test_plan_micro_batches_respects_budget_and_covers_all():
    lengths = [(10, 20, 5), (100, 120, 50), (12, 25, 7), (90, 110, 60), (5, 8, 1)]
    mbs = plan_micro_batches(lengths, max_tokens=200)
    assert sorted(i for mb in mbs for i in mb) == list(range(5))
    for mb in mbs:
        P = max(max(lengths[i][0], lengths[i][1]) for i in mb)
        L = max(lengths[i][2] for i in mb)
        assert len(mb) == 1 or len(mb) * (P + L) <= 200
    assert plan_micro_batches([(1000, 1000, 1000)], max_tokens=10) == [[0]]


@pytest.fixture(scope="module")
def tiny_llama():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=128)
    cfg._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(cfg).eval()


def test_padded_logits_match_unpadded(tiny_llama):
    ctx = [[5, 6, 7, 8, 9], [11, 12], [20, 21, 22]]
    resp = [[30, 31], [32, 33, 34, 35], [36]]
    b = build_aligned_batch(ctx, resp, pad_id=0)
    with torch.no_grad():
        logits = tiny_llama(input_ids=b.input_ids, attention_mask=b.attention_mask,
                            position_ids=b.position_ids).logits[:, b.logit_positions(), :]
        for r, (c, y) in enumerate(zip(ctx, resp)):
            ref = tiny_llama(input_ids=torch.tensor([c + y])).logits[0]
            want = ref[len(c) - 1: len(c) - 1 + len(y)]
            got = logits[r, : len(y)]
            assert torch.allclose(got, want, atol=1e-4), r

"""RP-OPD loss: top-K forward KL (Eq. 4) and the k3 reverse-KL ablation (App. B.4)."""

import pytest
import torch
import torch.nn.functional as F

from rrh.opd.loss import reverse_kl_k3, teacher_topk, topk_forward_kl


def _rand(n=6, v=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, v, generator=g) * 2, torch.randn(n, v, generator=g) * 2


def test_full_vocab_topk_equals_exact_forward_kl():
    t, s = _rand()
    idx, logp, _ = teacher_topk(t, k=t.shape[-1])
    loss, st = topk_forward_kl(s, idx, logp, torch.ones(t.shape[0]))
    pt, ls, lt = F.softmax(t, -1), F.log_softmax(s, -1), F.log_softmax(t, -1)
    exact = (pt * (lt - ls)).sum()
    assert loss.item() == pytest.approx(exact.item(), rel=1e-5)
    assert st["teacher_topk_mass_sum"] == pytest.approx(t.shape[0], rel=1e-5)


def test_zero_when_student_equals_teacher():
    t, _ = _rand()
    idx, logp, _ = teacher_topk(t, k=t.shape[-1])
    loss, st = topk_forward_kl(t.clone(), idx, logp, torch.ones(t.shape[0]))
    assert abs(loss.item()) < 1e-5
    assert st["top1_agree_sum"] == t.shape[0]


def test_topk_truncation_uses_unrenormalized_teacher_probs():
    t, s = _rand(n=3, v=40)
    k = 5
    idx, logp, _ = teacher_topk(t, k=k)
    assert idx.shape == (3, k) and torch.all(logp[:, :-1] >= logp[:, 1:])     # sorted desc
    loss, _ = topk_forward_kl(s, idx, logp, torch.ones(3))
    lt, ls = F.log_softmax(t, -1), F.log_softmax(s, -1)
    manual = sum((lt[i, idx[i]].exp() * (lt[i, idx[i]] - ls[i, idx[i]])).sum() for i in range(3))
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


def test_gradient_matches_analytic_form():
    # d l / d z_s = (sum_k p_T(k)) * softmax(z_s) - p_T restricted to top-K
    t, s = _rand(n=1, v=30)
    s = s.clone().requires_grad_(True)
    idx, logp, _ = teacher_topk(t, k=7)
    loss, _ = topk_forward_kl(s, idx, logp, torch.ones(1))
    loss.backward()
    pt = torch.zeros(30)
    pt[idx[0]] = logp[0].exp()
    expected = pt.sum() * F.softmax(s.detach()[0], -1) - pt
    assert torch.allclose(s.grad[0], expected, atol=1e-6)


def test_mask_excludes_padding_and_token_mean_is_split_invariant():
    t, s = _rand(n=8, v=20)
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1.])
    idx, logp, _ = teacher_topk(t, k=8)
    full, st = topk_forward_kl(s, idx, logp, mask)
    a, _ = topk_forward_kl(s[:3], idx[:3], logp[:3], mask[:3])
    b, _ = topk_forward_kl(s[3:], idx[3:], logp[3:], mask[3:])
    n = mask.sum()
    assert (full / n).item() == pytest.approx(((a + b) / n).item(), rel=1e-6)
    assert st["n_tokens"] == 6
    # padded rows contribute nothing even with garbage logits
    s2 = s.clone()
    s2[3] = 1e3
    again, _ = topk_forward_kl(s2, idx, logp, mask)
    assert again.item() == pytest.approx(full.item(), rel=1e-6)


def test_reverse_kl_k3_properties():
    t, s = _rand(n=5, v=25)
    ids = torch.tensor([0, 3, 7, 1, 2])
    lt = F.log_softmax(t, -1).gather(-1, ids[:, None])[:, 0]
    zero, _ = reverse_kl_k3(t.clone(), ids, lt, torch.ones(5))
    assert abs(zero.item()) < 1e-6
    s = s.clone().requires_grad_(True)
    loss, _ = reverse_kl_k3(s, ids, lt, torch.ones(5))
    assert loss.item() > 0
    loss.backward()
    assert s.grad.abs().sum() > 0


def test_teacher_topk_returns_sampled_logprob():
    t, _ = _rand(n=4, v=12)
    ids = torch.tensor([1, 2, 3, 4])
    _, _, samp = teacher_topk(t, 3, ids)
    assert torch.allclose(samp, F.log_softmax(t, -1).gather(-1, ids[:, None])[:, 0])

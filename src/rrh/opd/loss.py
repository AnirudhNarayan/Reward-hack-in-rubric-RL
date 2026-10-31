"""RP-OPD losses.

Main objective (Sec. 3, Eq. 4): forward KL(pi_T || pi_theta) at every student-sampled response
position, computed over the teacher's top-K tokens (K = 256):

    l_t = sum_{k in TopK_T(t)} p_T(k) * (log p_T(k) - log p_theta(k))

p_T is the teacher's full-softmax probability (not renormalized over the top-K, D2) and
log p_theta is the student's full-vocabulary log-softmax, so the gradient wrt the student logits
is (sum_k p_T(k)) * p_theta - p_T restricted to the top-K.

Ablation (App. B.4): reverse KL with the k3 estimator on the sampled token, backpropagated
through the student log-probability (veRL OPD "k3" / low_var_kl).

All functions take *flattened response positions*: logits [N, V], mask [N]; they return the
masked SUM over tokens so the caller can normalize by the global token count (token-mean, D3).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def teacher_topk(teacher_logits: torch.Tensor, k: int, sampled_ids: torch.Tensor | None = None):
    """-> (topk_idx [N,k] long, topk_logp [N,k] fp32, sampled_logp [N] fp32 | None)."""
    logp = F.log_softmax(teacher_logits.float(), dim=-1)
    vals, idx = logp.topk(min(k, logp.shape[-1]), dim=-1)
    sampled = logp.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1) if sampled_ids is not None else None
    return idx, vals, sampled


def topk_forward_kl(student_logits: torch.Tensor, topk_idx: torch.Tensor, topk_logp: torch.Tensor,
                    mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Masked sum over tokens of the top-K forward KL. Returns (loss_sum, stats_sums)."""
    s_logp_full = F.log_softmax(student_logits.float(), dim=-1)
    s_logp = s_logp_full.gather(-1, topk_idx)
    p_t = topk_logp.exp()
    per_tok = (p_t * (topk_logp - s_logp)).sum(-1)
    m = mask.to(per_tok.dtype)
    loss_sum = (per_tok * m).sum()
    stats = _token_stats(s_logp_full.detach(), topk_idx, p_t, m)
    stats["kl_sum"] = float(loss_sum.detach())
    return loss_sum, stats


def reverse_kl_k3(student_logits: torch.Tensor, sampled_ids: torch.Tensor,
                  teacher_sampled_logp: torch.Tensor, mask: torch.Tensor,
                  clamp: float = 10.0) -> tuple[torch.Tensor, dict]:
    """k3 = r - 1 - log r with r = p_T(y_t)/p_theta(y_t); gradient flows through log p_theta."""
    s_logp_full = F.log_softmax(student_logits.float(), dim=-1)
    s_lp = s_logp_full.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
    log_r = (teacher_sampled_logp - s_lp).clamp(-clamp, clamp)
    k3 = (log_r.exp() - 1.0 - log_r).clamp(-clamp, clamp)
    m = mask.to(k3.dtype)
    loss_sum = (k3 * m).sum()
    stats = {"kl_sum": float(loss_sum.detach()), "n_tokens": float(m.sum())}
    with torch.no_grad():
        stats.update(_entropy_sums(s_logp_full.detach(), m))
    return loss_sum, stats


@torch.no_grad()
def _entropy_sums(s_logp_full: torch.Tensor, m: torch.Tensor) -> dict:
    p = s_logp_full.exp()
    ent = -(p * s_logp_full).sum(-1)
    top20 = s_logp_full.topk(20, dim=-1).values
    q = F.softmax(top20, dim=-1)                       # renormalized over top-20 (Fig. 12b)
    ent20 = -(q * q.clamp_min(1e-30).log()).sum(-1)
    return {"entropy_sum": float((ent * m).sum()), "entropy_top20_sum": float((ent20 * m).sum())}


@torch.no_grad()
def _token_stats(s_logp_full, topk_idx, p_t, m) -> dict:
    out = _entropy_sums(s_logp_full, m)
    s_top1 = s_logp_full.argmax(-1)
    out.update({
        "n_tokens": float(m.sum()),
        "teacher_topk_mass_sum": float((p_t.sum(-1) * m).sum()),
        "student_mass_on_teacher_topk_sum": float((s_logp_full.gather(-1, topk_idx).exp().sum(-1) * m).sum()),
        "top1_agree_sum": float(((s_top1 == topk_idx[:, 0]).to(m.dtype) * m).sum()),
    })
    return out


def compute_loss(mode: str, student_logits, mask, *, topk_idx=None, topk_logp=None,
                 sampled_ids=None, teacher_sampled_logp=None):
    if mode == "forward_kl_topk":
        return topk_forward_kl(student_logits, topk_idx, topk_logp, mask)
    if mode == "reverse_kl_k3":
        return reverse_kl_k3(student_logits, sampled_ids, teacher_sampled_logp, mask)
    raise ValueError(f"unknown loss mode {mode!r}")

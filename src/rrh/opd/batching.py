"""Aligned padding so every response token sits in the same column for all rows.

Row layout (P = max context length, L = max response length in the micro-batch):

    [pad ... pad | ctx_1 .. ctx_n | y_1 .. y_m | pad ... pad]
     <------------ P ------------> <--------- L ---------->

The logits at columns P-1 .. P+L-2 predict y_1 .. y_L, so ``logits_to_keep`` can be one slice and
student and teacher (whose contexts differ in length) yield response logits of the same shape
[B, L, V]. position_ids restart at 0 on each row's first real token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class AlignedBatch:
    input_ids: torch.Tensor       # [B, P+L]
    attention_mask: torch.Tensor  # [B, P+L]
    position_ids: torch.Tensor    # [B, P+L]
    resp_ids: torch.Tensor        # [B, L]  (pad where resp_mask == 0)
    resp_mask: torch.Tensor       # [B, L]
    ctx_len: int                  # P

    def logit_positions(self) -> torch.Tensor:
        L = self.resp_ids.shape[1]
        return torch.arange(self.ctx_len - 1, self.ctx_len - 1 + L, device=self.input_ids.device)

    def to(self, device) -> "AlignedBatch":
        return AlignedBatch(self.input_ids.to(device), self.attention_mask.to(device),
                            self.position_ids.to(device), self.resp_ids.to(device),
                            self.resp_mask.to(device), self.ctx_len)


def build_aligned_batch(ctx_ids: Sequence[Sequence[int]], resp_ids: Sequence[Sequence[int]],
                        pad_id: int, resp_len: int | None = None) -> AlignedBatch:
    assert len(ctx_ids) == len(resp_ids) and len(ctx_ids) > 0
    P = max(len(c) for c in ctx_ids)
    L = resp_len or max(max(len(r) for r in resp_ids), 1)
    B = len(ctx_ids)
    ids = torch.full((B, P + L), pad_id, dtype=torch.long)
    att = torch.zeros((B, P + L), dtype=torch.long)
    r_ids = torch.full((B, L), pad_id, dtype=torch.long)
    r_mask = torch.zeros((B, L), dtype=torch.long)
    for b, (c, r) in enumerate(zip(ctx_ids, resp_ids)):
        r = list(r)[:L]
        ids[b, P - len(c):P] = torch.tensor(c, dtype=torch.long)
        att[b, P - len(c):P] = 1
        if r:
            ids[b, P:P + len(r)] = torch.tensor(r, dtype=torch.long)
            att[b, P:P + len(r)] = 1
            r_ids[b, :len(r)] = torch.tensor(r, dtype=torch.long)
            r_mask[b, :len(r)] = 1
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    return AlignedBatch(ids, att, pos, r_ids, r_mask, P)


def plan_micro_batches(lengths: Sequence[tuple[int, int, int]], max_tokens: int,
                       max_rows: int | None = None) -> list[list[int]]:
    """Greedy packing of sample indices into micro-batches.

    lengths[i] = (student_ctx_len, teacher_ctx_len, resp_len). The padded cost of a micro-batch
    is rows * (max ctx + max resp) on the longer (teacher) side. Samples are sorted by total
    length so padding stays small. A single over-budget sample still gets its own micro-batch.
    """
    order = sorted(range(len(lengths)), key=lambda i: max(lengths[i][0], lengths[i][1]) + lengths[i][2])
    out: list[list[int]] = []
    cur: list[int] = []
    cur_p = cur_l = 0
    for i in order:
        p = max(lengths[i][0], lengths[i][1])
        l = lengths[i][2]
        np_, nl = max(cur_p, p), max(cur_l, l)
        if cur and ((len(cur) + 1) * (np_ + nl) > max_tokens or (max_rows and len(cur) >= max_rows)):
            out.append(cur)
            cur, np_, nl = [], p, l
        cur.append(i)
        cur_p, cur_l = np_, nl
    if cur:
        out.append(cur)
    return out

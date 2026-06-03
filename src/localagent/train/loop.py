"""Shared training utilities: padding/collation, cosine LR, one AdamW step.

Kept tiny and dependency-free so pretrain/sft/grpo all reuse it (Phase 2/4/10).
"""

from __future__ import annotations

import math

import torch

from localagent.data.render import IGNORE


def pad_batch(rows: list[tuple[list[int], list[int]]], pad_id: int, device):
    """rows of (input_ids, labels) -> padded (x, y) tensors. y padded with IGNORE."""
    maxlen = max(len(r[0]) for r in rows)
    xs, ys = [], []
    for ids, labels in rows:
        n = maxlen - len(ids)
        xs.append(ids + [pad_id] * n)
        ys.append(labels + [IGNORE] * n)
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    # next-token shift: predict y[t+1] from x[t]
    return x[:, :-1], y[:, 1:]


def cosine_lr(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr

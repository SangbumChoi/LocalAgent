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


def wsd_lr(step: int, total: int, peak: float, warmup: int, decay_frac: float,
           min_ratio: float = 0.0) -> float:
    """Warmup-Stable-Decay LR (MiniCPM, 2404.06395).

    Three phases over `total` steps:
      - warmup  : linear 0 -> peak over the first `warmup` steps.
      - stable  : constant `peak` plateau (the reusable/forkable checkpoint lives here).
      - decay   : last `decay_frac` of steps, exponential `peak * 0.5^((s-S)/T)` where S is the
                  first decay step and T is the decay window length; the sharp loss drop happens
                  here. Clamped at `peak * min_ratio` (default 0 -> decays toward 0).

    With `decay_frac=0` this reduces to warmup + flat plateau (no decay).
    """
    if step < warmup:
        return peak * step / max(1, warmup)
    decay_steps = int(round(total * decay_frac))
    decay_start = total - decay_steps                      # S: first step of the decay window
    if step < decay_start or decay_steps <= 0:
        return peak                                        # stable plateau
    T = decay_steps                                        # decay time-constant = window length
    lr = peak * 0.5 ** ((step - decay_start) / max(1, T))
    return max(lr, peak * min_ratio)


def in_decay_window(step: int, total: int, decay_frac: float) -> bool:
    """True iff `step` falls in the WSD decay window (used to swap in curated `decay_samples`)."""
    decay_steps = int(round(total * decay_frac))
    return decay_steps > 0 and step >= total - decay_steps


def set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr

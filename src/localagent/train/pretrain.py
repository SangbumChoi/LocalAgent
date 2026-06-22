"""Pretrain from scratch (Phase 2, implemented): next-token CE over a packed byte stream.

Learns the byte distribution + conversation format before SFT focuses on the assistant span.
AdamW + cosine LR. Runs on GPU or CPU via the same code (train/device.py).
"""

from __future__ import annotations

import random

import torch

from localagent.train.device import Amp, resolve_dtype
from localagent.train.loop import cosine_lr, set_lr, wsd_lr


def pretrain(model, stream, tok, *, steps=400, batch_size=32, seq_len=128, lr=3e-3,
             warmup=30, device="cpu", log=print, lr_schedule="cosine", decay_frac=0.2,
             amp=False, amp_dtype="auto", log_every=None):
    """Next-token CE over a packed byte stream. `lr_schedule="wsd"` (opt-in, MiniCPM) replaces the
    cosine LR with Warmup-Stable-Decay (warmup -> flat plateau -> exponential `lr*0.5^((s-S)/T)`
    over the last `decay_frac` of steps); default "cosine" is byte-for-byte the old schedule.

    `amp=True` runs the forward/backward in mixed precision (bf16 on Ampere+, fp16 on older GPUs,
    or `amp_dtype` to force one). Default `amp=False` keeps the exact fp32 path used on CPU."""
    if lr_schedule not in ("cosine", "wsd"):
        raise ValueError(f"pretrain() lr_schedule must be 'cosine' or 'wsd', got {lr_schedule!r}")
    model.train()
    model.to(device)
    dev = torch.device(device) if isinstance(device, str) else device
    a = Amp(dev, resolve_dtype(dev, amp_dtype) if amp else torch.float32, enabled=amp)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    data = torch.tensor(stream, dtype=torch.long)
    n = data.numel()
    assert n > seq_len + 1, "pretrain stream too short"
    rng = random.Random(0)
    hist = []
    for step in range(steps):
        if lr_schedule == "wsd":
            set_lr(opt, wsd_lr(step, steps, lr, warmup, decay_frac, min_ratio=0.0))
        else:
            set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        starts = [rng.randint(0, n - seq_len - 2) for _ in range(batch_size)]
        batch = torch.stack([data[s:s + seq_len + 1] for s in starts]).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        with a.autocast():
            _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        a.backward(loss)
        a.step(opt, model.parameters())
        hist.append(loss.item())
        if step % (log_every or max(1, steps // 8)) == 0 or step == steps - 1:
            log(f"  [pretrain] step {step:4d}/{steps}  loss {loss.item():.3f}")
    return hist


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/flywheel.py — pretrain() is called in-process there")

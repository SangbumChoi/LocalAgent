"""Supervised fine-tune on agent samples (Phase 4, implemented).

Loss is masked to the assistant body + EOS (render.render_sft); the model only learns to produce
tool calls / text given the prompt, not to echo the user. Function masking (Hammer) is a TODO
hook — the deterministic templates already force copy-generalization to held-out slots.
"""

from __future__ import annotations

import random

import torch

from localagent.data.render import render_sft
from localagent.train.loop import cosine_lr, pad_batch, set_lr


def sft(model, samples, tok, *, steps=1200, batch_size=32, lr=1e-3, warmup=40,
        device="cpu", log=print):
    model.train()
    model.to(device)
    rows = [render_sft(s, tok) for s in samples]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        batch = [rng.choice(rows) for _ in range(batch_size)]
        x, y = pad_batch(batch, tok.pad_id, device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            log(f"  [sft] step {step:4d}/{steps}  loss {loss.item():.3f}")
    return hist


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/flywheel.py — sft() is called in-process there")

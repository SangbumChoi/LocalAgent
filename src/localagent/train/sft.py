"""Supervised fine-tune on agent samples (Phase 4, implemented).

Loss is masked to the assistant body + EOS (render.render_sft); the model only learns to produce
tool calls / text given the prompt, not to echo the user. Function masking (Hammer) is a TODO
hook — the deterministic templates already force copy-generalization to held-out slots.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from localagent.data.render import render_sft
from localagent.train.loop import cosine_lr, pad_batch, set_lr


def _framed_features(model, tok, prompts, device):
    """Last-prompt-token features for a batch of framed prompts (with grad), for the tool head."""
    from localagent.model.tokenizer import ASSISTANT, USER
    enc = [tok.encode(f"{USER}{p}{ASSISTANT}") for p in prompts]
    maxlen = max(len(e) for e in enc)
    X = torch.full((len(enc), maxlen), tok.pad_id, dtype=torch.long, device=device)
    last = []
    for i, e in enumerate(enc):
        X[i, : len(e)] = torch.tensor(e, device=device)
        last.append(len(e) - 1)
    _, feats = model(X, return_hidden=True)
    return feats[torch.arange(len(enc)), torch.tensor(last, device=device)]


def sft(model, samples, tok, *, steps=1200, batch_size=32, lr=1e-3, warmup=40,
        device="cpu", log=print, joint_tool_head=False, aux_weight=1.0):
    """SFT with masked LM loss. With `joint_tool_head`, also trains a tool-selection head from an
    auxiliary classification loss so the model's features become tool-discriminative (dual-head).
    Returns (loss_hist, head) where head is None unless joint_tool_head."""
    model.train()
    model.to(device)
    rows = [render_sft(s, tok) for s in samples]
    head, labels = None, None
    params = list(model.parameters())
    if joint_tool_head:
        from localagent.agent.tool_head import CLASSES, ToolHead, label_of
        head = ToolHead(model.cfg.d_model).to(device)
        labels = [CLASSES.index(label_of(s)) for s in samples]
        params += list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        idx = [rng.randrange(len(rows)) for _ in range(batch_size)]
        x, y = pad_batch([rows[i] for i in idx], tok.pad_id, device)
        _, loss = model(x, targets=y)
        if joint_tool_head:
            feat = _framed_features(model, tok, [samples[i].prompt for i in idx], device)
            lab = torch.tensor([labels[i] for i in idx], device=device)
            loss = loss + aux_weight * F.cross_entropy(head(feat), lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            log(f"  [sft] step {step:4d}/{steps}  loss {loss.item():.3f}")
    return hist, head


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/flywheel.py — sft() is called in-process there")

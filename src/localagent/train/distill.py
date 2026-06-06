"""Distillation (Phase 6, implemented): offline logit KD from a larger teacher into the student.

Both models are byte-level (vocab 256), so we distill the **next-byte distribution** on the agent
data. The teacher's soft targets are cached once (batched), then the student trains against them —
no teacher forward in the inner loop, which makes it affordable on CPU.

  forward_kl : KL(teacher || student)  — standard KD (match the teacher everywhere).
  reverse_kl : KL(student || teacher)  — mode-seeking (MiniLLM); good for generation.

The KD is applied on the assistant/tool-call spans (where the label mask is active), optionally
mixed with a little ground-truth CE.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from localagent.data.render import IGNORE, render_sft
from localagent.train.loop import cosine_lr, set_lr


@torch.no_grad()
def cache_teacher(teacher, rows, tok, device="cpu", temperature=2.0, batch_size=16, log=print):
    """Cache teacher log-probs (at `temperature`) for each row's next-token positions."""
    teacher.eval().to(device)
    cache = [None] * len(rows)
    for i in range(0, len(rows), batch_size):
        seqs = [rows[j][0][:-1] for j in range(i, min(i + batch_size, len(rows)))]
        ml = max(len(s) for s in seqs)
        X = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            X[r, : len(s)] = torch.tensor(s, device=device)
        logits, _ = teacher(X)
        lp = (logits / temperature).log_softmax(-1)
        for r, s in enumerate(seqs):
            cache[i + r] = lp[r, : len(s)].half().cpu()   # (len, V)
        if i % (batch_size * 10) == 0:
            log(f"  [cache] {i}/{len(rows)}")
    return cache


def distill(student, samples, teacher, tok, *, steps=400, batch_size=24, kd_type="forward_kl",
            temperature=2.0, kd_weight=1.0, ce_weight=0.1, lr=1e-3, warmup=30, device="cpu",
            log=print):
    """Distill `teacher` (a trained model) into `student` via cached logit KD on agent samples."""
    rows = [render_sft(s, tok) for s in samples]
    log("caching teacher soft targets ...")
    cache = cache_teacher(teacher, rows, tok, device, temperature, log=log)
    student.train().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95))
    rng = random.Random(0)
    V = student.cfg.vocab_size
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        bi = [rng.randrange(len(rows)) for _ in range(batch_size)]
        seqs = [rows[j][0][:-1] for j in bi]
        labs = [rows[j][1][1:] for j in bi]
        ml = max(len(s) for s in seqs)
        X = torch.full((len(bi), ml), tok.pad_id, dtype=torch.long, device=device)
        Y = torch.full((len(bi), ml), IGNORE, dtype=torch.long, device=device)
        TL = torch.zeros(len(bi), ml, V, device=device)
        for r, j in enumerate(bi):
            X[r, : len(seqs[r])] = torch.tensor(seqs[r], device=device)
            Y[r, : len(labs[r])] = torch.tensor(labs[r], device=device)
            tc = cache[j]
            TL[r, : tc.shape[0]] = tc.float().to(device)
        logits, _ = student(X)
        s_lp = (logits / temperature).log_softmax(-1)
        mask = (Y != IGNORE).float()
        if kd_type == "reverse_kl":
            kd = (s_lp.exp() * (s_lp - TL)).sum(-1)
        else:  # forward_kl
            kd = (TL.exp() * (TL - s_lp)).sum(-1)
        kd = (kd * mask).sum() / mask.sum().clamp(min=1) * (temperature ** 2)
        loss = kd_weight * kd
        if ce_weight > 0:
            loss = loss + ce_weight * F.cross_entropy(
                logits.reshape(-1, V), Y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(f"  [distill] step {step}/{steps}  loss {loss.item():.3f}")
    return hist


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/distill_demo.py — distill() is called in-process there")

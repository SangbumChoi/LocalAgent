"""Distillation (Phase 6, implemented): offline logit KD from a larger teacher into the student.

Both models are byte-level (vocab 256), so we distill the **next-byte distribution** on the agent
data. The teacher's soft targets are cached once (batched), then the student trains against them —
no teacher forward in the inner loop, which makes it affordable on CPU.

  forward_kl : KL(teacher || student)  — standard KD (match the teacher everywhere).
  reverse_kl : KL(student || teacher)  — mode-seeking (MiniLLM); good for generation.
  topk       : LFM2 "decoupled, tempered Top-K KD" — only cache the teacher's Top-K tokens
               plus the off-Top-K tail mass, then match (a) the renormalized distribution over
               the Top-K support and (b) the in-vs-out-of-support mass split. Avoids the
               support-mismatch instability of full-vocab KD and keeps the cache to K/pos.

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


@torch.no_grad()
def cache_teacher_topk(teacher, rows, tok, device="cpu", temperature=2.0, k=16, batch_size=16,
                       log=print):
    """Cache the teacher's tempered Top-K targets per next-token position (LFM2 KD).

    For each position we store, in a compact per-row dict of tensors:
      ids  : (len, K) long  — the teacher's Top-K token ids (by tempered prob)
      p    : (len, K) half  — teacher tempered probs at those ids (softmax over the FULL vocab)
      tail : (len,)   half  — 1 - sum(top-K probs), the mass outside the Top-K support
    K is clamped to the vocab size, so for byte models (vocab 256) it can be the full vocab.
    """
    teacher.eval().to(device)
    V = teacher.cfg.vocab_size
    k = min(k, V)
    cache = [None] * len(rows)
    for i in range(0, len(rows), batch_size):
        seqs = [rows[j][0][:-1] for j in range(i, min(i + batch_size, len(rows)))]
        ml = max(len(s) for s in seqs)
        X = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            X[r, : len(s)] = torch.tensor(s, device=device)
        logits, _ = teacher(X)
        probs = (logits / temperature).softmax(-1)            # (B, ml, V) over FULL vocab
        topp, topi = probs.topk(k, dim=-1)                    # (B, ml, K)
        tail = (1.0 - topp.sum(-1)).clamp(min=0.0)            # (B, ml)
        for r, s in enumerate(seqs):
            n = len(s)
            cache[i + r] = {
                "ids": topi[r, :n].cpu(),
                "p": topp[r, :n].half().cpu(),
                "tail": tail[r, :n].half().cpu(),
            }
        if i % (batch_size * 10) == 0:
            log(f"  [cache-topk] {i}/{len(rows)}")
    return cache


def _topk_kd_loss(logits, cache, bi, X, mask, V, temperature, device):
    """Decoupled tempered Top-K KD term (LFM2), summed over masked positions then averaged.

    Per position p, with teacher Top-K support S (ids), teacher tempered probs t_S over S, and
    teacher tail mass m_tail = 1 - sum(t_S):
      KL_topk = KL(t_hat_S || s_hat_S), where t_hat / s_hat renormalize t_S / s_S over S only.
      tail_kl = binary KL( [sum(t_S), m_tail]  ||  [sum(s_S), 1 - sum(s_S)] ).
    Returns (KL_topk + tail_kl) * T^2 averaged over masked positions.
    """
    B, ml = X.shape
    k = cache[bi[0]]["ids"].shape[1]
    ids = torch.zeros(B, ml, k, dtype=torch.long, device=device)
    t_p = torch.zeros(B, ml, k, device=device)
    t_tail = torch.zeros(B, ml, device=device)
    for r, j in enumerate(bi):
        c = cache[j]
        n = c["ids"].shape[0]
        ids[r, :n] = c["ids"].to(device)
        t_p[r, :n] = c["p"].float().to(device)
        t_tail[r, :n] = c["tail"].float().to(device)
    s_probs = (logits / temperature).softmax(-1)              # (B, ml, V)
    s_p = s_probs.gather(-1, ids)                             # (B, ml, K) student probs on S
    eps = 1e-9
    # (1) Top-K matching: renormalize both over S, KL(teacher_S || student_S).
    t_in = t_p.sum(-1).clamp(min=eps)                         # teacher mass on S
    s_in = s_p.sum(-1).clamp(min=eps)                         # student mass on S
    t_hat = t_p / t_in.unsqueeze(-1)
    s_hat = s_p / s_in.unsqueeze(-1)
    kl_topk = (t_hat * ((t_hat + eps).log() - (s_hat + eps).log())).sum(-1)
    # (2) Tail-mass matching: binary KL between (in_S, out_S) splits. The teacher's out-of-S
    # mass is the cached `tail` (= 1 - sum(top-K)); the student's is computed live from s_in.
    t_in_c = (1.0 - t_tail).clamp(min=eps, max=1.0)           # teacher in-S mass (from cache)
    t_out = t_tail.clamp(min=eps, max=1.0)
    s_in_c = s_in.clamp(min=eps, max=1.0)
    s_out = (1.0 - s_in_c).clamp(min=eps)
    tail_kl = (t_in_c * ((t_in_c + eps).log() - (s_in_c + eps).log())
               + t_out * ((t_out + eps).log() - (s_out + eps).log()))
    kd = (kl_topk + tail_kl) * mask
    return kd.sum() / mask.sum().clamp(min=1) * (temperature ** 2)


def distill(student, samples, teacher, tok, *, steps=400, batch_size=24, kd_type="forward_kl",
            temperature=2.0, kd_weight=1.0, ce_weight=0.1, kd_k=16, lr=1e-3, warmup=30,
            device="cpu", log=print):
    """Distill `teacher` (a trained model) into `student` via cached logit KD on agent samples."""
    rows = [render_sft(s, tok) for s in samples]
    is_topk = kd_type == "topk"
    log("caching teacher soft targets ..." + (" (top-k)" if is_topk else ""))
    if is_topk:
        cache = cache_teacher_topk(teacher, rows, tok, device, temperature, k=kd_k, log=log)
    else:
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
        for r, j in enumerate(bi):
            X[r, : len(seqs[r])] = torch.tensor(seqs[r], device=device)
            Y[r, : len(labs[r])] = torch.tensor(labs[r], device=device)
        logits, _ = student(X)
        mask = (Y != IGNORE).float()
        if is_topk:
            kd = _topk_kd_loss(logits, cache, bi, X, mask, V, temperature, device)
        else:
            TL = torch.zeros(len(bi), ml, V, device=device)
            for r, j in enumerate(bi):
                tc = cache[j]
                TL[r, : tc.shape[0]] = tc.float().to(device)
            s_lp = (logits / temperature).log_softmax(-1)
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

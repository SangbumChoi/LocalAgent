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


def _framed_full(model, tok, prompts, device):
    """Per-token features for a batch of framed prompts (with grad). Returns
    (feats (B,Tmax,d), lengths (B,), framed_ids list)."""
    from localagent.model.tokenizer import ASSISTANT, USER
    enc = [tok.encode(f"{USER}{p}{ASSISTANT}") for p in prompts]
    maxlen = max(len(e) for e in enc)
    X = torch.full((len(enc), maxlen), tok.pad_id, dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        X[i, : len(e)] = torch.tensor(e, device=device)
    _, feats = model(X, return_hidden=True)
    lengths = torch.tensor([len(e) for e in enc], device=device)
    return feats, lengths, enc


def sft(model, samples, tok, *, steps=1200, batch_size=32, lr=1e-3, warmup=40,
        device="cpu", log=print, joint_tool_head=False, aux_weight=1.0, conversations=None):
    """SFT with masked LM loss over single-turn samples + optional multi-turn `conversations`
    (which teach tool->response->follow-up continuation). With `joint_tool_head`, also trains
    jointly a tool-selection head AND a pointer/copy argument head (on the single-turn samples).
    Returns (loss_hist, tool_head, ptr_head); heads are None unless joint_tool_head."""
    import json

    from localagent.data.render import render_conversation
    model.train()
    model.to(device)
    rows = [render_sft(s, tok) for s in samples]
    lm_rows = rows + ([render_conversation(c, tok) for c in (conversations or [])])
    tool_head = ptr_head = None
    params = list(model.parameters())
    meta = None
    if joint_tool_head:
        from localagent.agent.pointer_head import ARG_IDX, PointerHead, gold_span
        from localagent.agent.tool_head import CLASSES, ToolHead, label_of
        tool_head = ToolHead(model.cfg.d_model).to(device)
        ptr_head = PointerHead(model.cfg.d_model).to(device)
        params += list(tool_head.parameters()) + list(ptr_head.parameters())
        meta = []  # (tool_label, ptr_arg_name|None, ptr_value_ids|None)
        for s in samples:
            parg = pval = None
            if s.kind == "tool":
                for k, v in json.loads(s.ref_args).items():
                    if k in ARG_IDX:
                        parg, pval = k, tok.encode(v)
                        break
            meta.append((CLASSES.index(label_of(s)), parg, pval))
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        idx_lm = [rng.randrange(len(lm_rows)) for _ in range(batch_size)]
        x, y = pad_batch([lm_rows[i] for i in idx_lm], tok.pad_id, device)
        _, loss = model(x, targets=y)
        if joint_tool_head:
            from localagent.agent.pointer_head import ARG_IDX, gold_span
            idx = [rng.randrange(len(samples)) for _ in range(batch_size)]  # heads: single-turn
            feats, lengths, enc = _framed_full(model, tok, [samples[i].prompt for i in idx], device)
            last = feats[torch.arange(len(idx)), lengths - 1]
            loss = loss + aux_weight * F.cross_entropy(
                tool_head(last), torch.tensor([meta[i][0] for i in idx], device=device))
            # pointer head: rows in the batch that have a copy arg with a locatable gold span
            rws, gs, ge, ai = [], [], [], []
            for bi, i in enumerate(idx):
                _, parg, pval = meta[i]
                if parg is None:
                    continue
                span = gold_span(enc[bi], pval)
                if span is None:
                    continue
                rws.append(bi); gs.append(span[0]); ge.append(span[1]); ai.append(ARG_IDX[parg])
            if rws:
                sub = feats[rws]                                   # (k,Tmax,d)
                sl, el = ptr_head.logits(sub, torch.tensor(ai, device=device))
                for r, bi in enumerate(rws):                       # mask padding positions
                    sl[r, lengths[bi]:] = -1e9; el[r, lengths[bi]:] = -1e9
                loss = loss + aux_weight * (
                    F.cross_entropy(sl, torch.tensor(gs, device=device))
                    + F.cross_entropy(el, torch.tensor(ge, device=device)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            log(f"  [sft] step {step:4d}/{steps}  loss {loss.item():.3f}")
    return hist, tool_head, ptr_head


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/flywheel.py — sft() is called in-process there")

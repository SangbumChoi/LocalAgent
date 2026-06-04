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
        device="cpu", log=print, joint_tool_head=False, aux_weight=1.0, ptr_weight=0.15,
        conversations=None):
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
        meta = []  # (tool_label|-1, ptr_arg_name|None, ptr_value_ids|None); -1 = skip (parallel)
        for s in samples:
            if s.calls:                       # parallel: skip head training (handled by split)
                meta.append((-1, None, None))
                continue
            parg = pval = None
            if s.kind == "tool":
                for k, v in json.loads(s.ref_args).items():
                    if k in ARG_IDX:
                        parg, pval = k, tok.encode(v)
                        break
            meta.append((CLASSES.index(label_of(s)), parg, pval))
        # multi-turn head examples: train tool+pointer heads on episode contexts too, so they
        # transfer to multi-turn (the tool head reads a context ending after a tool response;
        # the pointer can copy a follow-up arg out of that response).
        mt = []  # (ctx_ids, tool_label, ptr_arg_idx|-1, gold_start, gold_end)
        from localagent.data.render import history_text
        from localagent.model.tokenizer import ASSISTANT
        for conv in (conversations or []):
            for i, msg in enumerate(conv.messages):
                if msg.role.value != "assistant" or not msg.tool_calls:
                    continue
                c = msg.tool_calls[0]
                cid = tok.encode(history_text(conv.messages[:i]) + ASSISTANT)
                lab = CLASSES.index(c.name) if c.name in CLASSES else CLASSES.index("text")
                pa, gsx, gex = -1, -1, -1
                for k, v in c.arguments.items():
                    if k in ARG_IDX and (sp := gold_span(cid, tok.encode(v))):
                        pa, gsx, gex = ARG_IDX[k], sp[0], sp[1]
                        break
                mt.append((cid, lab, pa, gsx, gex))
        single_idx = [i for i, s in enumerate(samples) if not s.calls]  # head-trainable samples
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
            idx = [rng.choice(single_idx) for _ in range(batch_size)]  # heads: single-call only
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
                loss = loss + ptr_weight * (   # down-weighted so it can't swamp tool selection
                    F.cross_entropy(sl, torch.tensor(gs, device=device))
                    + F.cross_entropy(el, torch.tensor(ge, device=device)))
            # --- multi-turn head training (episode contexts) ---
            if mt:
                mb = [mt[rng.randrange(len(mt))] for _ in range(min(12, len(mt)))]
                ml = max(len(e[0]) for e in mb)
                X = torch.full((len(mb), ml), tok.pad_id, dtype=torch.long, device=device)
                for r, e in enumerate(mb):
                    X[r, : len(e[0])] = torch.tensor(e[0], device=device)
                _, mfeats = model(X, return_hidden=True)
                mlast = mfeats[torch.arange(len(mb)), torch.tensor([len(e[0]) - 1 for e in mb])]
                loss = loss + aux_weight * F.cross_entropy(
                    tool_head(mlast), torch.tensor([e[1] for e in mb], device=device))
                prw = [r for r, e in enumerate(mb) if e[2] >= 0]
                if prw:
                    sl, el = ptr_head.logits(mfeats[prw], torch.tensor([mb[r][2] for r in prw],
                                                                       device=device))
                    for j, r in enumerate(prw):
                        sl[j, len(mb[r][0]):] = -1e9; el[j, len(mb[r][0]):] = -1e9
                    loss = loss + ptr_weight * (
                        F.cross_entropy(sl, torch.tensor([mb[r][3] for r in prw], device=device))
                        + F.cross_entropy(el, torch.tensor([mb[r][4] for r in prw], device=device)))
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

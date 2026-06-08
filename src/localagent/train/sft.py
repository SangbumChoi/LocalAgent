"""Supervised fine-tune on agent samples (Phase 4, implemented).

Loss is masked to the assistant body + EOS (render.render_sft); the model only learns to produce
tool calls / text given the prompt, not to echo the user. Function masking (Hammer) is a TODO
hook — the deterministic templates already force copy-generalization to held-out slots.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from localagent.data.render import IGNORE, render_sft
from localagent.train.loop import cosine_lr, in_decay_window, pad_batch, set_lr, wsd_lr


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
        conversations=None, accum_steps=1, mt_weight=1.0,
        teacher=None, kd_type="topk", kd_k=16, kd_weight=0.5, kd_temperature=2.0,
        lr_schedule="cosine", decay_frac=0.2, decay_samples=None, shuffle=True):
    """SFT with masked LM loss over single-turn samples + optional multi-turn `conversations`
    (which teach tool->response->follow-up continuation). With `joint_tool_head`, also trains
    jointly a tool-selection head AND a pointer/copy argument head (on the single-turn samples).

    `steps` is the number of OPTIMIZER steps; each runs `accum_steps` micro-batches of size
    `batch_size` (effective batch = batch_size * accum_steps). Each micro-batch's combined loss is
    divided by `accum_steps` and backward()'d immediately, so peak memory stays at one micro-batch
    regardless of accumulation. `mt_weight` scales ONLY the multi-turn head-training losses (tool +
    pointer CE on episode contexts); the LM loss on rendered conversations is unaffected.

    **distill-throughout-SFT** (optional, default OFF): if `teacher` is given, the teacher's
    Top-K next-token targets are cached ONCE on the single-turn SFT `samples` (reusing
    distill.py's `cache_teacher_topk`, memory = K/pos not full vocab), and each step adds
    `kd_weight * topk_kd_loss(student_logits, teacher_topk)` on the assistant spans alongside the
    LM/head/pointer losses. The backbone keeps matching the teacher's distribution WHILE the heads
    train, so it is not pulled away from verbatim arg-copying. Only `kd_type="topk"` is supported
    here (it reuses distill.py's `_topk_kd_loss`). When `teacher is None` the path is inert and
    every existing caller is byte-for-byte unchanged.

    **WSD schedule** (opt-in, MiniCPM 2404.06395): `lr_schedule="wsd"` switches the per-step LR
    from cosine to Warmup-Stable-Decay — linear warmup -> flat `lr` plateau -> exponential
    `lr*0.5^((s-S)/T)` over the last `decay_frac` of steps (T = decay-window length). Default
    `lr_schedule="cosine"` is byte-for-byte the old schedule. `decay_samples` (a separate, ideally
    cleaner/curated sample pool) is OPTIONAL: when given AND on WSD, the single-turn LM rows drawn
    during the decay window come from `decay_samples` instead of the main pool — the on-device
    "inject your cleanest data in the decay window" trick. Multi-turn `conversations` and the head
    items are unchanged (heads keep their full training distribution).

    **Ordered (curriculum) passes** (opt-in): with `shuffle=False`, the single-turn LM micro-batches
    are drawn by walking `lm_rows` *in their given order* (a contiguous cursor, wrapping at the end)
    instead of i.i.d. uniform sampling. Feed pre-ordered samples (e.g. easy->hard via
    ``agent_synth.curriculum_order``) and the model sees easy rows first, hard rows later — the
    LFM2-style curriculum. Default `shuffle=True` is byte-for-byte the old i.i.d. behaviour. Only the
    LM stream is affected; head / pointer / multi-turn / KD micro-batches stay i.i.d. (their job is
    coverage, not ordering).
    Returns (loss_hist, tool_head, ptr_head); heads are None unless joint_tool_head."""
    import json

    from localagent.data.render import render_conversation
    model.train()
    model.to(device)
    if lr_schedule not in ("cosine", "wsd"):
        raise ValueError(f"sft() lr_schedule must be 'cosine' or 'wsd', got {lr_schedule!r}")
    rows = [render_sft(s, tok) for s in samples]
    conv_rows = [render_conversation(c, tok) for c in (conversations or [])]
    lm_rows = rows + conv_rows
    # WSD decay-window data injection: during the decay phase, draw single-turn LM rows from the
    # (cleaner) `decay_samples` pool instead of `rows`. conv_rows always come along so multi-turn
    # continuation coverage is never dropped. Inert unless lr_schedule=="wsd" and decay_samples set.
    decay_lm_rows = lm_rows
    if lr_schedule == "wsd" and decay_samples is not None:
        decay_lm_rows = [render_sft(s, tok) for s in decay_samples] + conv_rows

    # --- distill-throughout-SFT: cache teacher Top-K targets ONCE on the SFT rows ---
    kd_cache = None
    if teacher is not None:
        if kd_type != "topk":
            raise ValueError("sft() distillation only supports kd_type='topk'")
        from localagent.train.distill import _topk_kd_loss, cache_teacher_topk
        log(f"  [sft] caching teacher top-{kd_k} targets for distill-throughout-SFT ...")
        kd_cache = cache_teacher_topk(teacher, rows, tok, device=device,
                                      temperature=kd_temperature, k=kd_k, log=log)
        V_kd = model.cfg.vocab_size
    tool_head = ptr_head = None
    params = list(model.parameters())
    meta = None
    if joint_tool_head:
        from localagent.agent.pointer_head import ARG_IDX, PointerHead, gold_span
        from localagent.agent.tool_head import CLASSES, ToolHead, label_of
        tool_head = ToolHead(model.cfg.d_model).to(device)
        ptr_head = PointerHead(model.cfg.d_model).to(device)
        params += list(tool_head.parameters()) + list(ptr_head.parameters())
        # Head training items: (prompt, tool_label, ptr_arg_name|None, ptr_value_ids|None).
        # Parallel "X and Y" turns are SPLIT into conjuncts so the head learns the (lowercased)
        # fragments the parallel decoder will actually feed it.
        head_items = []

        def _ptr_of(args):
            for k, v in args.items():
                if k in ARG_IDX:
                    return k, tok.encode(v)
            return None, None

        for s in samples:
            if s.calls:                                   # parallel -> one item per conjunct
                conj = s.prompt.split(" and ")
                if len(conj) == len(s.calls):
                    for cpr, call in zip(conj, s.calls):
                        lab = CLASSES.index(call["name"]) if call["name"] in CLASSES \
                            else CLASSES.index("text")
                        pa, pv = _ptr_of(call["arguments"])
                        head_items.append((cpr.strip(), lab, pa, pv))
            else:
                args = json.loads(s.ref_args) if s.kind == "tool" else {}
                pa, pv = _ptr_of(args)
                head_items.append((s.prompt, CLASSES.index(label_of(s)), pa, pv))
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
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    hist = []
    lm_cursor = [0]  # mutable cell: contiguous read position for ordered (shuffle=False) passes

    def _next_lm_idx(lm_pool):
        """`batch_size` LM-row indices into `lm_pool`. Shuffled => i.i.d. uniform (old behaviour);
        ordered (shuffle=False) => the next contiguous block, wrapping, so a pre-ordered (e.g.
        easy->hard curriculum) pool is consumed in order across steps."""
        if shuffle:
            return [rng.randrange(len(lm_pool)) for _ in range(batch_size)]
        start = lm_cursor[0]
        idx = [(start + j) % len(lm_pool) for j in range(batch_size)]
        lm_cursor[0] = (start + batch_size) % len(lm_pool)
        return idx

    def _micro_loss(lm_pool):
        """Full combined loss (LM + head + ptr + mt) for ONE micro-batch of `batch_size`.
        `lm_pool` is the LM-row pool to sample (swapped to curated rows in the WSD decay window)."""
        idx_lm = _next_lm_idx(lm_pool)
        x, y = pad_batch([lm_pool[i] for i in idx_lm], tok.pad_id, device)
        _, loss = model(x, targets=y)
        if kd_cache is not None:
            # KD micro-batch sampled from the SFT rows (the only rows with cached teacher
            # targets), batched exactly like distill.py: inputs = row[:-1], mask on labels[1:].
            bi = [rng.randrange(len(rows)) for _ in range(batch_size)]
            seqs = [rows[j][0][:-1] for j in bi]
            labs = [rows[j][1][1:] for j in bi]
            ml = max(len(s) for s in seqs)
            Xk = torch.full((len(bi), ml), tok.pad_id, dtype=torch.long, device=device)
            mk = torch.zeros(len(bi), ml, device=device)
            for r in range(len(bi)):
                Xk[r, : len(seqs[r])] = torch.tensor(seqs[r], device=device)
                lab_t = torch.tensor(labs[r], device=device)
                mk[r, : len(lab_t)] = (lab_t != IGNORE).float()
            klogits, _ = model(Xk)
            kd = _topk_kd_loss(klogits, kd_cache, bi, Xk, mk, V_kd, kd_temperature, device)
            loss = loss + kd_weight * kd
        if joint_tool_head:
            from localagent.agent.pointer_head import ARG_IDX, gold_span
            batch = [head_items[rng.randrange(len(head_items))] for _ in range(batch_size)]
            feats, lengths, enc = _framed_full(model, tok, [b[0] for b in batch], device)
            last = feats[torch.arange(len(batch)), lengths - 1]
            loss = loss + aux_weight * F.cross_entropy(
                tool_head(last), torch.tensor([b[1] for b in batch], device=device))
            # pointer head: rows in the batch that have a copy arg with a locatable gold span
            rws, gs, ge, ai = [], [], [], []
            for bi, (_, _, parg, pval) in enumerate(batch):
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
            # --- multi-turn head training (episode contexts), scaled by mt_weight ---
            if mt:
                mb = [mt[rng.randrange(len(mt))] for _ in range(min(12, len(mt)))]
                ml = max(len(e[0]) for e in mb)
                X = torch.full((len(mb), ml), tok.pad_id, dtype=torch.long, device=device)
                for r, e in enumerate(mb):
                    X[r, : len(e[0])] = torch.tensor(e[0], device=device)
                _, mfeats = model(X, return_hidden=True)
                mlast = mfeats[torch.arange(len(mb)), torch.tensor([len(e[0]) - 1 for e in mb])]
                loss = loss + mt_weight * aux_weight * F.cross_entropy(
                    tool_head(mlast), torch.tensor([e[1] for e in mb], device=device))
                prw = [r for r, e in enumerate(mb) if e[2] >= 0]
                if prw:
                    sl, el = ptr_head.logits(mfeats[prw], torch.tensor([mb[r][2] for r in prw],
                                                                       device=device))
                    for j, r in enumerate(prw):
                        sl[j, len(mb[r][0]):] = -1e9; el[j, len(mb[r][0]):] = -1e9
                    loss = loss + mt_weight * ptr_weight * (
                        F.cross_entropy(sl, torch.tensor([mb[r][3] for r in prw], device=device))
                        + F.cross_entropy(el, torch.tensor([mb[r][4] for r in prw], device=device)))
        return loss

    for step in range(steps):
        if lr_schedule == "wsd":
            set_lr(opt, wsd_lr(step, steps, lr, warmup, decay_frac, min_ratio=0.0))
            lm_pool = decay_lm_rows if in_decay_window(step, steps, decay_frac) else lm_rows
        else:
            set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
            lm_pool = lm_rows
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum_steps):
            loss = _micro_loss(lm_pool) / accum_steps
            loss.backward()                  # free this micro-batch's graph before the next forward
            step_loss += loss.item() * accum_steps
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(step_loss)
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            log(f"  [sft] step {step:4d}/{steps}  loss {step_loss:.3f}")
    return hist, tool_head, ptr_head


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/flywheel.py — sft() is called in-process there")

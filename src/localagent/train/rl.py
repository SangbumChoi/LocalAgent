"""GRPO RL fine-tuning (Phase 10, implemented) with a *verifiable* reward.

For each prompt we sample G rollouts, score each with the eval correctness check (reward ∈ {0,1}
— exactly the metric we optimize, no learned reward model needed), compute group-relative
advantages, and do a policy-gradient step on the rollout token log-probs. This is the RL stage
that pushes a near-but-not-100% SFT model the rest of the way.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from localagent.data.render import prompt_text
from localagent.eval.harness import _correct
from localagent.train.device import Amp, resolve_dtype
from localagent.train.loop import cosine_lr, set_lr


@torch.no_grad()
def _rollout(model, tok, prompt_ids, max_new, temperature, device):
    ids = list(prompt_ids)
    caches = [None] * model.n_cache_slots()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _, caches = model(x, pos=0, caches=caches)
    pos = len(ids)
    gen = []
    for _ in range(max_new):
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        if nxt == tok.eos_id:
            break
        gen.append(nxt)
        step = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, _, caches = model(step, pos=pos, caches=caches)
        pos += 1
    return gen


def _logprob_sum(model, prompt_ids, gen_ids, device):
    """Sum log p(gen | prompt) under the current model (with grad)."""
    full = torch.tensor([prompt_ids + gen_ids], dtype=torch.long, device=device)
    logits, _ = model(full[:, :-1])
    logp = F.log_softmax(logits[0], dim=-1)
    targets = full[0, 1:]
    tok_lp = logp[torch.arange(targets.shape[0]), targets]
    start = len(prompt_ids) - 1  # first position predicting a gen token
    return tok_lp[start:].sum()


def grpo(model, samples, tok, *, steps=60, prompts_per_step=8, group_size=4, lr=2e-4,
         temperature=1.0, max_new=64, device="cpu", log=print, amp=False, amp_dtype="auto"):
    model.to(device)
    dev = torch.device(device) if isinstance(device, str) else device
    amp_h = Amp(dev, resolve_dtype(dev, amp_dtype) if amp else torch.float32, enabled=amp)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    rng = random.Random(0)
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, 5, 0.1))
        batch = [rng.choice(samples) for _ in range(prompts_per_step)]
        opt.zero_grad(set_to_none=True)
        rewards_log, n_updated = [], 0
        # Backward per prompt-group (not over all 48 rollouts at once) to bound memory.
        for s in batch:
            pid = tok.encode(prompt_text(s))
            model.eval()
            rollouts = [_rollout(model, tok, pid, max_new, temperature, device)
                        for _ in range(group_size)]
            rewards = torch.tensor(
                [float(_correct(s, tok.decode(r))) for r in rollouts], device=device)
            rewards_log.append(rewards.mean().item())
            if rewards.std() < 1e-6:
                continue  # no signal in this group
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            model.train()
            with amp_h.autocast():
                gl = [(-adv_i * _logprob_sum(model, pid, r, device))
                      for r, adv_i in zip(rollouts, adv) if r]
                g_loss = torch.stack(gl).mean() / len(batch) if gl else None
            if g_loss is not None:
                amp_h.backward(g_loss)                            # accumulate grads, free graph
                n_updated += 1
        if n_updated:
            amp_h.step(opt, model.parameters())
        avg_r = sum(rewards_log) / max(1, len(rewards_log))
        hist.append(avg_r)
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(f"  [grpo] step {step:3d}/{steps}  mean_reward {avg_r:.3f}")
    return hist


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/flywheel.py — grpo() is called in-process there")

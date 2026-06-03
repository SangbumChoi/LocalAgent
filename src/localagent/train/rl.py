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


def grpo(model, samples, tok, *, steps=60, prompts_per_step=8, group_size=6, lr=2e-4,
         temperature=1.0, max_new=96, device="cpu", log=print):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    rng = random.Random(0)
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, 5, 0.1))
        batch = [rng.choice(samples) for _ in range(prompts_per_step)]
        losses, rewards_log = [], []
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
            for r, a in zip(rollouts, adv):
                if not r:
                    continue
                lp = _logprob_sum(model, pid, r, device)
                losses.append(-a * lp)
        if not losses:
            hist.append(sum(rewards_log) / max(1, len(rewards_log)))
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        avg_r = sum(rewards_log) / max(1, len(rewards_log))
        hist.append(avg_r)
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(f"  [grpo] step {step:3d}/{steps}  mean_reward {avg_r:.3f}")
    return hist


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/flywheel.py — grpo() is called in-process there")

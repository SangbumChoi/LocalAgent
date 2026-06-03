#!/usr/bin/env python
"""Data-flywheel driver for the ultra-tiny (~1M, byte-level) agent.

Per round: SFT (+ short GRPO) on the current enrichment level, evaluate on HELD-OUT slot values,
then enrich (level += 1) and repeat. The model persists across rounds (round 1 trains hardest;
later rounds adapt incrementally).

Two metrics, both honest:
  - free-gen  : the model autoregressively generates the call (the raw ability).
  - grounded  : prompt-grounded constrained decoding (the deployed decoder; ARCHITECTURE_IDEAS
                §2b) — the model ranks candidate calls whose args are grounded in the prompt.
A <100M byte model learns call *structure* fast but not generalizable slot *copying*, so free-gen
stays low on held-out while grounded decoding reaches ~100%. That gap is the whole point.

Outputs (runs/flywheel/): metrics.json, accuracy.png, freegen_vs_grounded.png, loss.png,
samples.json, ultra-tiny.pt

Usage:  python scripts/flywheel.py [--rounds 5] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from localagent.data.agent_synth import Generator
from localagent.data.render import build_pretrain_stream, prompt_text
from localagent.eval.harness import evaluate, evaluate_grounded
from localagent.inference.generate import generate
from localagent.agent.constrained import grounded_decode
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.pretrain import pretrain
from localagent.train.rl import grpo
from localagent.train.sft import sft

OUT = "runs/flywheel"


def fmt(d):
    return f"overall={d['overall']*100:.1f}%  " + " ".join(
        f"{k}={v*100:.0f}%" for k, v in sorted(d["groups"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    model = LocalAgentLM(cfg).to(device)
    print(f"model {cfg.name}: {model.num_params()/1e6:.3f}M params on {device}", flush=True)

    n_train = 400 if args.quick else 1200
    n_eval = 12 if args.quick else 30        # per category (balanced held-out)
    pre_steps = 60 if args.quick else 200
    sft1 = 150 if args.quick else 600
    sft_inc = 80 if args.quick else 250
    grpo_steps = 4 if args.quick else 12

    g0 = Generator(level=1, seed=0, split="train").generate(n_train)
    pre_loss = pretrain(model, build_pretrain_stream(g0, tok), tok, steps=pre_steps,
                        batch_size=64, device=device)

    metrics = {"rounds": [], "pretrain_loss": pre_loss}
    for r in range(1, args.rounds + 1):
        train = Generator(level=r, seed=r, split="train").generate(n_train)
        held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(n_eval)
        steps = sft1 if r == 1 else sft_inc
        print(f"\n=== Round {r} (level {r}, {len(train)} train / {len(held)} held-out) ===", flush=True)
        sft_loss, head = sft(model, train, tok, steps=steps, batch_size=32, lr=1.5e-3,
                             device=device, log=lambda *a: None, joint_tool_head=True)
        grpo(model, train, tok, steps=grpo_steps, device=device, log=lambda *a: None)  # RL stage
        gr = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head)
        print(f"  grounded (held-out): {fmt(gr)}", flush=True)
        metrics["rounds"].append({"round": r, "level": r, "grounded": gr,
                                  "sft_loss_last": sft_loss[-1]})
        torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, f"{OUT}/ultra-tiny.pt")
        json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
        _plot_rounds(metrics)

    # final comparison: raw free-generation vs grounded, on the last level's held-out
    final_held = Generator(level=args.rounds, seed=4242, split="eval").generate_balanced(20)
    fg = evaluate(model, final_held, tok, device=device)
    gr = evaluate_grounded(model, final_held, tok, TOOLS, device=device, tool_head=head)
    metrics["final_freegen"] = fg
    metrics["final_grounded"] = gr
    print(f"\nFINAL free-gen : {fmt(fg)}")
    print(f"FINAL grounded : {fmt(gr)}")

    samples = []
    for s in Generator(level=args.rounds, seed=999, split="eval").generate(8):
        out = grounded_decode(model, tok, s.prompt, TOOLS, device=device, tool_head=head)
        samples.append({"prompt": s.prompt, "expected": s.target, "grounded_out": out})
    json.dump(samples, open(f"{OUT}/samples.json", "w"), indent=2)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    _plot_rounds(metrics); _plot_compare(metrics); _plot_loss(pre_loss)
    print(f"\nArtifacts in {OUT}/ (accuracy.png, freegen_vs_grounded.png, loss.png, samples.json)")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_rounds(metrics):
    try:
        plt = _mpl()
    except Exception:
        return
    rs = [m["round"] for m in metrics["rounds"]]
    groups = sorted({g for m in metrics["rounds"] for g in m["grounded"]["groups"]})
    fig, ax = plt.subplots(figsize=(7, 4))
    for g in groups:
        ax.plot(rs, [m["grounded"]["groups"].get(g, 0) * 100 for m in metrics["rounds"]],
                marker="o", label=g)
    ax.plot(rs, [m["grounded"]["overall"] * 100 for m in metrics["rounds"]],
            marker="s", lw=2.5, color="black", label="overall")
    ax.set_xlabel("flywheel round (enrichment level)"); ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 105); ax.set_xticks(rs); ax.grid(alpha=.3); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ultra-tiny ~1M: grounded held-out accuracy across flywheel rounds")
    fig.tight_layout(); fig.savefig(f"{OUT}/accuracy.png", dpi=120); plt.close(fig)


def _plot_compare(metrics):
    try:
        plt = _mpl(); import numpy as np
    except Exception:
        return
    fg, gr = metrics["final_freegen"], metrics["final_grounded"]
    groups = sorted(gr["groups"])
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, [fg["groups"].get(g, 0) * 100 for g in groups], w, label="free-gen (raw)")
    ax.bar(x + w / 2, [gr["groups"].get(g, 0) * 100 for g in groups], w, label="grounded decode")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 105); ax.grid(alpha=.3, axis="y"); ax.legend()
    ax.set_title("Same 1M model: raw byte generation vs prompt-grounded decoding")
    fig.tight_layout(); fig.savefig(f"{OUT}/freegen_vs_grounded.png", dpi=120); plt.close(fig)


def _plot_loss(pre_loss):
    try:
        plt = _mpl()
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pre_loss); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("Pretrain loss (next-byte CE)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/loss.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()

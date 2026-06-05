#!/usr/bin/env python
"""Failure-driven data flywheel: generate -> test -> ANALYZE -> enrich the weak tools -> repeat 5x.

Unlike the level-bump flywheel, this mines each round's per-category eval and **oversamples the
categories the model is failing** in the next round's training data (weight = 1 + k*(1-acc)). The
analysis (weakest tools + new sampling weights) is printed and saved each round.

Single-turn only (15→21 tools) for speed, so the 5-round loop completes on CPU.
Outputs (runs/analyze/): analysis.json, analyze.png
Usage:  python scripts/analyze_loop.py [--rounds 5] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import REALISTIC_WEIGHTS, Generator
from localagent.data.render import build_pretrain_stream
from localagent.eval.harness import evaluate_grounded, multi_turn_eval
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.pretrain import pretrain
from localagent.train.sft import sft

OUT = "runs/analyze"
K = 3.0  # how aggressively to oversample weak categories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--model", default="configs/model/ultra-tiny-1m.yaml")
    ap.add_argument("--pre", type=int, default=0, help="pretrain steps (0=auto)")
    ap.add_argument("--sft1", type=int, default=0, help="round-1 SFT steps (0=auto)")
    ap.add_argument("--sft-inc", type=int, default=0, help="later-round SFT steps (0=auto)")
    ap.add_argument("--episodes", type=int, default=0,
                    help="multi-turn trajectory episodes per round to train on (0=auto)")
    ap.add_argument("--batch", type=int, default=32, help="SFT batch size (lower to fit memory)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    cfg = ModelConfig.from_yaml(args.model)
    global OUT
    OUT = f"runs/analyze_{cfg.name}"
    os.makedirs(OUT, exist_ok=True)
    model = LocalAgentLM(cfg).to(device)
    print(f"model {cfg.name}: {model.num_params()/1e6:.3f}M params on {device}", flush=True)

    n_train = 1200 if args.quick else 10000   # 4x larger
    n_eval = 8 if args.quick else 16
    n_ep = args.episodes or (40 if args.quick else 160)   # mixed coding/productivity/planner
    pre, s1, sinc = (40, 120, 60) if args.quick else (200, 300, 200)
    pre = args.pre or pre
    s1 = args.sft1 or s1
    sinc = args.sft_inc or sinc

    g0 = Generator(level=1, seed=0, split="train").generate(n_train)
    pretrain(model, build_pretrain_stream(g0, tok), tok, steps=pre, batch_size=64, device=device)

    weights = dict(REALISTIC_WEIGHTS)     # realistic base (parallel-heavy, calc down-weighted)
    hist = []
    best_overall = -1.0
    for r in range(1, args.rounds + 1):
        train = Generator(level=r, seed=r, split="train").generate_weighted(n_train, weights)
        # NEW: multi-turn trajectory episodes (coding + computer-use + planner-then-execute), so the
        # model learns the plan->act decomposition and follow-up args grounded in tool responses.
        episodes = Generator(level=r, seed=5000 + r, split="train").episodes(n_ep)
        held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(n_eval)
        held_ep = Generator(level=r, seed=6000 + r, split="eval").episodes(max(8, n_ep // 4))
        steps = s1 if r == 1 else sinc
        _, head, ptr = sft(model, train, tok, steps=steps, batch_size=args.batch, lr=1.5e-3,
                           device=device, log=lambda *a: None, joint_tool_head=True,
                           conversations=episodes)
        res = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
        mt = multi_turn_eval(model, held_ep, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
        cats = res["categories"]
        weak = sorted(cats.items(), key=lambda kv: kv[1])[:5]
        # ANALYZE -> reweight: failing categories oversampled next round, on the realistic base
        new_weights = {c: round(REALISTIC_WEIGHTS.get(c, 1.0) * (1 + K * (1 - a)), 2)
                       for c, a in cats.items()}
        print(f"\n=== Round {r}: overall={res['overall']*100:.1f}%  "
              f"(trained with {len(train)} single-turn + {len(episodes)} episodes) ===", flush=True)
        print("  weakest: " + ", ".join(f"{c}={a*100:.0f}%" for c, a in weak), flush=True)
        print(f"  multi-turn: step_acc={mt['step_acc']*100:.0f}% episode_acc={mt['episode_acc']*100:.0f}%"
              f" ({mt['steps']} steps)", flush=True)
        print("  -> next round oversamples: " + ", ".join(
            f"{c}x{new_weights[c]}" for c, _ in weak), flush=True)
        hist.append({"round": r, "overall": res["overall"], "categories": cats, "multi_turn": mt,
                     "weights_for_next": new_weights, "n_train": len(train), "n_episodes": len(episodes)})
        weights = new_weights
        # value parallel AND trajectory step-accuracy in the best-save score
        score = res["overall"] + 0.5 * cats.get("parallel", 0.0) + 0.3 * mt["step_acc"]
        if score > best_overall:     # keep the BEST round, not the last (can regress)
            best_overall = score
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
                        "tool_head": head.state_dict() if head is not None else None,
                        "ptr_head": ptr.state_dict() if ptr is not None else None},
                       f"{OUT}/model.pt")
            print(f"  (saved best model: score={best_overall*100:.1f}, "
                  f"overall={res['overall']*100:.1f}% mt_step={mt['step_acc']*100:.0f}%)", flush=True)
        json.dump(hist, open(f"{OUT}/analysis.json", "w"), indent=2)
        _plot(hist)

    print(f"\nDone. overall by round: {[round(h['overall']*100) for h in hist]}  "
          f"mt_step by round: {[round(h['multi_turn']['step_acc']*100) for h in hist]}  "
          f"(best score={best_overall*100:.1f} saved to {OUT}/model.pt)")


def _plot(hist):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [h["round"] for h in hist]
    # track the categories that were weakest in round 1, to show failure-driven recovery
    weak0 = sorted(hist[0]["categories"].items(), key=lambda kv: kv[1])[:5]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, [h["overall"] * 100 for h in hist], marker="s", lw=2.5, color="black", label="overall")
    if all("multi_turn" in h for h in hist):
        ax.plot(xs, [h["multi_turn"]["step_acc"] * 100 for h in hist], marker="D", ms=5, lw=2,
                color="tab:purple", label="multi-turn step_acc")
    for c, _ in weak0:
        ax.plot(xs, [h["categories"].get(c, 0) * 100 for h in hist], marker="o", ms=4,
                label=f"{c} (weak@R1)")
    ax.set_xlabel("flywheel round")
    ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax.set_title("Failure-driven flywheel: oversampling the weak tools each round")
    fig.tight_layout()
    fig.savefig(f"{OUT}/analyze.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

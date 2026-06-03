#!/usr/bin/env python
"""Data-flywheel driver: pretrain -> [SFT -> GRPO -> eval] per round, enrich, repeat.

Trains the ultra-tiny (~1M, byte-level) model from scratch and drives it toward 100% on the
agent categories (tool-calling, web-search, planner, text). Each round enriches the synthetic
dataset (level += 1). Eval uses held-out slot values (disjoint pools) so accuracy = generalization.

Outputs (runs/flywheel/):
  metrics.json   per-round group accuracies + train losses
  accuracy.png   accuracy per group across rounds
  loss.png       pretrain + SFT loss curves
  samples.json   example generations (visualize the output)
  ultra-tiny.pt  trained checkpoint

Usage:  python scripts/flywheel.py [--rounds 5] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from localagent.data.agent_synth import Generator
from localagent.data.render import build_pretrain_stream, prompt_text
from localagent.eval.harness import evaluate
from localagent.inference.generate import generate
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.pretrain import pretrain
from localagent.train.rl import grpo
from localagent.train.sft import sft

OUT = "runs/flywheel"


def fmt(d: dict) -> str:
    g = d["groups"]
    parts = " ".join(f"{k}={v*100:.0f}%" for k, v in sorted(g.items()))
    return f"overall={d['overall']*100:.1f}%  {parts}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="tiny run to smoke-test the chain")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    model = LocalAgentLM(cfg).to(device)
    print(f"model {cfg.name}: {model.num_params()/1e6:.3f}M params on {device}")

    n_train = 300 if args.quick else 1500
    n_eval = 100 if args.quick else 300
    pre_steps = 80 if args.quick else 350
    sft_steps = 200 if args.quick else 900
    grpo_steps = 10 if args.quick else 40

    # Pretrain once (from scratch) on round-1 data to learn bytes + format.
    g0 = Generator(level=1, seed=0, split="train").generate(n_train)
    stream = build_pretrain_stream(g0, tok)
    pre_loss = pretrain(model, stream, tok, steps=pre_steps, device=device)

    metrics = {"rounds": [], "pretrain_loss": pre_loss}
    for r in range(1, args.rounds + 1):
        level = r
        train = Generator(level=level, seed=r, split="train").generate(n_train)
        held = Generator(level=level, seed=1000 + r, split="eval").generate(n_eval)
        print(f"\n=== Round {r} (enrichment level {level}, {len(train)} train / {len(held)} eval) ===")

        sft_loss = sft(model, train, tok, steps=sft_steps, device=device)
        res = evaluate(model, held, tok, device=device)
        print(f"  after SFT : {fmt(res)}")

        if res["overall"] < 1.0:  # push the rest of the way with RL
            grpo(model, train, tok, steps=grpo_steps, device=device)
            res = evaluate(model, held, tok, device=device)
            print(f"  after GRPO: {fmt(res)}")

        metrics["rounds"].append({
            "round": r, "level": level, "eval": res,
            "sft_loss_last": sft_loss[-1],
        })
        torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()},
                   f"{OUT}/ultra-tiny.pt")
        # "regenerate to enrich" happens automatically next round via level += 1.

    # ---- sample outputs (visualize the output) ----
    samples = []
    demo = Generator(level=args.rounds, seed=999, split="eval").generate(8)
    for s in demo:
        gen, st = generate(model, tok, prompt_text(s), temperature=0.0)
        samples.append({"prompt": s.prompt, "expected": s.target, "got": gen,
                        "decode_tok_s": round(st.decode_tok_s, 1)})
    json.dump(samples, open(f"{OUT}/samples.json", "w"), indent=2)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)

    _plot(metrics, pre_loss)
    print("\nFinal:", fmt(metrics["rounds"][-1]["eval"]))
    print(f"Artifacts in {OUT}/  (accuracy.png, loss.png, samples.json, metrics.json)")


def _plot(metrics, pre_loss):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e} — skipping plots)")
        return
    rounds = [m["round"] for m in metrics["rounds"]]
    groups = sorted({g for m in metrics["rounds"] for g in m["eval"]["groups"]})
    fig, ax = plt.subplots(figsize=(7, 4))
    for g in groups:
        ax.plot(rounds, [m["eval"]["groups"].get(g, 0) * 100 for m in metrics["rounds"]],
                marker="o", label=g)
    ax.plot(rounds, [m["eval"]["overall"] * 100 for m in metrics["rounds"]],
            marker="s", linewidth=2.5, color="black", label="overall")
    ax.set_xlabel("flywheel round (enrichment level)"); ax.set_ylabel("held-out accuracy (%)")
    ax.set_title("LocalAgent ultra-tiny (~1M): accuracy per category across flywheel rounds")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/accuracy.png", dpi=120)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(pre_loss, label="pretrain", alpha=0.8)
    ax2.set_xlabel("step"); ax2.set_ylabel("loss"); ax2.set_title("Pretrain loss (next-byte CE)")
    ax2.grid(alpha=0.3); ax2.legend()
    fig2.tight_layout(); fig2.savefig(f"{OUT}/loss.png", dpi=120)


if __name__ == "__main__":
    main()

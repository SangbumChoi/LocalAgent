# CLAUDE.md

Project guidance for Claude Code. **Read `AGENTS.md` first** — it has the setup, build/test
commands, and conventions, and is the source of truth shared with Cursor and Codex. This file
adds Claude-specific sub-agent routing.

## TL;DR
Pure-PyTorch, <100M-param, from-scratch tool-calling agent. Three tiers (ultra-tiny ~1M byte-level,
tiny ~30M, small ~90M). Keep `pytest -q` green and `ruff check` clean. Never bypass the model
param-budget guard. Report real eval numbers — no faking 100%.

## Sub-agents (delegate by area)
Specialized sub-agents live in `.claude/agents/`. Prefer delegating focused work to them so each
keeps a tight context. Route like this:

| When the task is about… | Use sub-agent |
|---|---|
| synthetic data, templates, enrichment, the `Conversation` schema, the flywheel | `data-engineer` |
| the model architecture, training loops (pretrain/SFT/GRPO), the KV cache, optimizers | `model-trainer` |
| eval harness, AST tool scoring, accuracy/regression checks, benchmarks | `evaluator` |
| export to GGUF/ONNX/ExecuTorch, quantization, parity, on-device perf | `exporter` |

The main thread stays the orchestrator: it plans, wires stages in `scripts/flywheel.py` /
`pipeline/flow.py`, and integrates sub-agent results. Sub-agents own *their* module and must keep
the shared contracts (the `Conversation` schema, config YAMLs, the budget guard) intact — they
control their own part, not the cross-cutting interfaces.

## Guardrails
- Do not add heavy ML frameworks (`transformers`, `trl`, `deepspeed`, …). Pure PyTorch.
- Do not edit the `Conversation` schema or a model config without saying so explicitly — these are
  cross-cutting contracts other sub-agents depend on.
- Training/benchmark runs are CPU-friendly but can be slow; use `--quick` for smoke checks and run
  long jobs in the background.
- Artifacts (`runs/`, checkpoints, `*.png`) are git-ignored; surface PNGs to the user instead of
  committing them.

## Useful commands
See `AGENTS.md`. Quickest signal: `pytest -q` then `python scripts/flywheel.py --quick`.

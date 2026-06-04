# AGENTS.md — guide for coding agents (Codex, Cursor, Claude Code, …)

This is the cross-tool instructions file (the `AGENTS.md` convention). Claude Code also reads
`CLAUDE.md`, which defers to this file and adds sub-agent routing. Cursor rules live in
`.cursor/rules/`. Keep all three in sync when conventions change.

## What this project is
LocalAgent trains a **< 100M-param** LLM (tiers: ultra-tiny ~1M, tiny ~30M, small ~90M) **from
scratch** in **pure PyTorch** to act as a local tool-calling **agent**. The ultra-tiny tier is
**byte-level** (vocab 256). See `docs/ARCHITECTURE.md`, `docs/ARCHITECTURE_IDEAS.md`,
`docs/ROADMAP.md`, `docs/RESEARCH.md`.

## Setup / build / test
```bash
pip install -e ".[dev,demo]"     # demo extra pulls matplotlib for the viz scripts
pytest -q                        # unit tests (config/model/schema/agent) — keep green
ruff check src tests             # lint
```

## Run the pipeline
```bash
localagent model-info configs/model/ultra-tiny-1m.yaml   # param budget report
python scripts/flywheel.py --quick                       # fast end-to-end smoke (CPU OK)
python scripts/flywheel.py --rounds 5                    # full train+enrich flywheel
python scripts/analyze_loop.py --rounds 5                # failure-driven flywheel (oversamples weak tools)
python scripts/benchmark.py                              # tok/s + memory viz (KV cache vs not)
python scripts/push_to_hf.py --checkpoint runs/flywheel/ultra-tiny.pt --out runs/hf_export
                                                         # build HF bundle (+ --repo <u>/<n> --push to upload)
```

## Conventions (please follow)
- **Pure PyTorch only.** No `transformers`/`trl`/training frameworks. A BPE tokenizer lib is the
  only model-adjacent dependency, and the ultra-tiny tier avoids even that (byte-level).
- **One interchange format:** `localagent.data.schema.Conversation`. Data flows synth → flywheel →
  train → eval through it. Add a field there, not in ad-hoc dicts.
- **Configs over flags:** model/train/data settings live in `configs/*.yaml`.
- **Budget guard:** every model config must pass `ModelConfig.assert_within_budget()` (<100M).
  Don't bypass it.
- **Honest stubs:** unimplemented code raises `NotImplementedError("TODO(phase-N): …")` pointing at
  `docs/ROADMAP.md`. Don't fake results; if eval isn't 100%, report the real number.
- **Determinism in data:** synthetic targets are canonical (sorted-key compact JSON) and train/eval
  slot pools are disjoint. Preserve both — they're what make exact-match eval meaningful.
- Style: ruff, line length 100, type hints on public functions, match surrounding code.

## Map of the code
| Area | Path |
|---|---|
| Model (decoder, KV cache, config+budget, tokenizer) | `src/localagent/model/` |
| Data (schema, synth generator, render, flywheel) | `src/localagent/data/` |
| Training (pretrain, sft, grpo, device, loop utils) | `src/localagent/train/` |
| Eval (AST tool eval, harness) | `src/localagent/eval/` |
| Agent (runtime loop, tools, parser, memory) | `src/localagent/agent/` |
| Inference (KV-cache generate, export) | `src/localagent/inference/` |
| Orchestration | `src/localagent/pipeline/`, `scripts/flywheel.py` |

## Before you commit
- `pytest -q` green, `ruff check` clean.
- If you changed a model config, paste the `localagent model-info` line in the PR/commit.
- Don't commit `runs/`, `data/`, checkpoints, `*.png` artifacts (see `.gitignore`).

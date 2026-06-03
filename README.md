# LocalAgent

A **< 100M-parameter** LLM, **pretrained from scratch**, that works as a local **agent** —
it does **tool calling** and **text generation**. Built as a **minimal, hackable, pure-PyTorch**
ML pipeline (nanochat in spirit), with **training + evaluation**, that runs on **CPU / GPU / NPU**
and exports to **PyTorch / GGUF / ONNX / ExecuTorch**. A **data flywheel** lets it improve from
the conversations it has while running locally.

> Status: **Phase 0 — scaffold.** The architecture, configs, schema, model, and CLI skeleton are
> in place; per-stage logic lands phase by phase. See the status table below.

## Why / what
- **Tiny + from scratch** — you build and pretrain the transformer yourself; a budget guard keeps
  every config under 100M params.
- **Agent-native** — tool calls are *in-vocabulary* (`<tool_call>…</tool_call>`), so the model
  natively decides between answering, calling a tool, or abstaining.
- **One codebase, no framework** — pure PyTorch + a BPE tokenizer + YAML configs. Every stage is a
  readable script you can fork.
- **Runs anywhere** — one device abstraction (CUDA/MPS/XPU/CPU) for training; four export targets
  for inference on CPU/GPU/NPU.
- **Gets better with use** — served conversations are stored, mined, verified, and fed back in.

The design synthesizes the best ideas from prior work — see [`docs/RESEARCH.md`](docs/RESEARCH.md)
for what we take from nanochat, SmolLM2, APIGen/xLAM, ToolACE, Hammer, MiniLLM/on-policy-
distillation, MemGPT/Letta, the Airbnb data flywheel, and BFCL — but it isn't bound by them:
[`docs/ARCHITECTURE_IDEAS.md`](docs/ARCHITECTURE_IDEAS.md) lays out the original structural bets
(agent-as-controller, the vocabulary tax, factorized + depth-recurrent ultra-tiny models, a dual
tool/text head, grammar-constrained decoding). [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is
how it all fits together.

### Three size tiers
| Tier | ~params | Job | Vocab | Structure |
|---|---|---|---|---|
| `ultra-tiny-1m` | ~1.0M | **tool router / planner** (picks tool + args, abstains) | byte 256 | factorized embeddings + depth-recurrence (2 blocks × 6 loops = effective depth 12) |
| `tiny-30m` | ~31M | **agent** (tool calls + short grounded answers) | BPE 32k | standard decoder |
| `small-90m` | ~89M | **capable local agent** (multi-turn tool use + coherent text) | BPE 32k | standard decoder |

At ~1M params the goal is a *controller*, not a chatbot: knowledge is offloaded to tools and the
memory/retrieval store, so the parameters can encode skills and control flow.

## Quickstart
```bash
pip install -e .

# inspect a model config and check it's under the 100M budget
localagent model-info configs/model/tiny-30m.yaml

# run the toy end-to-end pipeline (stubs no-op until their phase is built)
bash scripts/speedrun.sh

pytest        # core (config/model/schema/agent) tests pass today
```

## Repo layout
```
configs/                YAML for every model/train/data stage
docs/                   RESEARCH.md · ARCHITECTURE.md · ARCHITECTURE_IDEAS.md · ROADMAP.md
src/localagent/
  model/                tiny Llama-style decoder (GQA+RoPE+SwiGLU), config+budget, tokenizer
  data/                 Conversation schema · pretrain corpus · agent_synth · flywheel
  train/                pretrain · sft · distill · rl · device abstraction
  eval/                 AST tool eval (BFCL-style) · text eval · harness
  agent/                runtime loop · tool registry · tool-call parser · two-tier memory
  inference/            KV-cached generate · export/{gguf,onnx,executorch}
  pipeline/             stage DAG orchestration
  cli.py                `localagent ...`
demos/                  chat_cli.py · web/app.py (visualized agent loop)
scripts/                speedrun.sh · download_data.py
tests/
```

## Build status
| Area | State |
|---|---|
| Model: 3 tiers (decoder, config, budget guard) + factorized embeddings + depth-recurrence | ✅ implemented (Phase 1 core) |
| Dual tool/text head + grammar-constrained decoding | 📐 proposed (ARCHITECTURE_IDEAS.md, Phase 4) |
| Conversation schema, tool registry, tool-call parser, AST eval primitives | ✅ implemented |
| Device abstraction (CPU/GPU/NPU) | ✅ implemented |
| Tokenizer training | 🚧 stub (Phase 1) |
| Pretrain / SFT / Distill / RL loops | 🚧 stubs (Phases 2/4/6/10) |
| Agent data synthesis + flywheel | 🚧 stubs (Phases 3/8) |
| Eval harness (multi-turn, parity) | 🚧 stubs (Phase 5) |
| Agent runtime + memory + demos | 🚧 stubs (Phase 7) |
| Export GGUF/ONNX/ExecuTorch | 🚧 stubs (Phase 9) |

Full plan: [`docs/ROADMAP.md`](docs/ROADMAP.md). Stubs raise `NotImplementedError` with a
`TODO(phase-N)` pointing at the roadmap, so nothing silently pretends to work.

## License
MIT

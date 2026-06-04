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

### Result (ultra-tiny ~1M, pretrained from scratch, byte-level)
A **21-tool** agent — general (`get_weather`, `calculator`, `web_search`, `planner`, `define`,
`play_music`, `get_news`), the **Claude Code / Codex-style coding surface** (`read_file`,
`write_file`, `grep_search`, `run_command`, `git_commit`, `run_tests`), **computer-use /
productivity** (`calendar_event`, `send_email`, `open_url` browser, `notion_write`, `slack_send`,
`jira_issue`), and everyday (`set_reminder`, `set_timer`). On **held-out slot values** (disjoint
train/eval pools), the raw byte model learns tool-call *structure* perfectly but can't copy unseen
slot values. The deployed decoder is a **dual head + prompt-grounded constrained decoding**: a
jointly-trained tool-selection head picks the tool, and arguments are grounded in spans of the
prompt (schema-driven, trigger-free) — a learned **pointer/copy head** for the general/multi-turn
case, exact heuristic extractors for clean single-turn (verified 0 misses across 1,616 args).

| decoder | overall (15 tools, held-out) |
|---|---|
| raw byte generation | ~0% |
| **dual-head + grounded constrained** | **~83%** (text/define/web_search 100%, news 92%, planner 83%, code 80%, calc/weather 71%, music 73%) |

The data flywheel (5 enrichment rounds) lifts the harder categories as it grows — `code` 46%→80%,
`news` 25%→92%. Argument grounding is exact (0/1254 extraction misses across all tools); remaining
errors are tool *selection* by the 1M head (e.g. news↔weather). See `runs/flywheel/accuracy.png`.

**Multi-turn coding episodes + pointer head.** Training the tool head and a learned
**pointer/copy argument head** (`agent/pointer_head.py`) jointly — including on multi-turn episode
contexts — lets the agent run tool→response→follow-up trajectories and ground a follow-up arg
(e.g. a file path) out of an earlier **tool response** (which heuristics can't reach). In a
combined 5-round run, multi-turn coding episodes score **~73–77% per step / ~50% whole-episode**
(see `runs/flywheel/final_combined.png`). Sharing the SFT step budget with the extra heads lowers
single-turn peak (~67% here vs ~83% single-turn-only) — a compute trade-off, not a method limit.
On clean single-turn templates the deterministic extractors still beat the pointer head, so
single-turn deploys heuristics and multi-turn uses the pointer head.

**Scaling the tool surface (21 tools).** Adding the computer-use/productivity tools brings the set
to 21 (a 22-way selector). Grounding stays exact (0/1616), but single-turn *selection* accuracy
drops to ~47% — distinctive-argument tools are reliable (`send_email`→a name, `open_url`→a URL),
while tools that share an argument *shape* (`calendar_event` / `notion_write` / `slack_send` /
`jira_issue` all take a quoted string) confuse the 1M tool head even though their values extract
correctly. More tools ⇒ harder selection on a 1M head; a larger tier (`tiny-30m`) or
per-tool-balanced training disambiguates. This is a selection-capacity limit, not a grounding one.

Throughput/memory (CPU, 4 threads), showing the KV-cache win:

| tier | params | prefill tok/s | decode (KV cache) | decode (no cache) | speedup | param mem |
|---|---|---|---|---|---|---|
| ultra-tiny-1m | 0.98M | 5400 | 177 tok/s | 63 | ×2.8 | 3.9 MB |
| tiny-30m | 31M | 1834 | 96 tok/s | 24 | ×4.0 | 125 MB |
| small-90m | 89M | 831 | 47 tok/s | 11 | ×4.2 | 357 MB |

Reproduce: `python scripts/flywheel.py` (train+enrich loop → `runs/flywheel/accuracy.png`),
`python scripts/benchmark.py` (→ `runs/bench/throughput.png`, `memory.png`).

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
| Model: 3 tiers (decoder, **KV cache**, config, budget guard) + factorized embeddings + depth-recurrence | ✅ implemented |
| Training: pretrain + SFT + GRPO (verifiable reward), CPU/GPU | ✅ implemented |
| Synthetic data + render + eval harness (AST + grounded) | ✅ implemented |
| **Dual-head (tool classifier) + grounded constrained decoding** → ~83% held-out across 15 tools | ✅ implemented (`agent/constrained.py`, `agent/tool_head.py`) |
| Pointer/copy argument head (learned span grounding) | ✅ implemented (`agent/pointer_head.py`) |
| Multi-turn coding episodes (tool→response→follow-up) + replay eval | ✅ implemented (`data/agent_synth.py`, `eval/harness.py`) |
| Flywheel driver + throughput/memory + accuracy visualizations | ✅ implemented (`scripts/flywheel.py`, `scripts/benchmark.py`) |
| Dual tool/text head, SSM backbone, retrieval head | 📐 proposed (ARCHITECTURE_IDEAS.md) |
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

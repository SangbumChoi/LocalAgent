# LocalAgent

A **< 100M-parameter** LLM, **pretrained from scratch**, that works as a local **agent** —
it does **tool calling** and **text generation**. Built as a **minimal, hackable, pure-PyTorch**
ML pipeline (nanochat in spirit), with **training + evaluation**, that runs on **CPU / GPU / NPU**
and exports to **PyTorch / GGUF / ONNX / ExecuTorch**. A **data flywheel** lets it improve from
the conversations it has while running locally.

> **🤗 Model · 🚀 Demo · 📦 Data — all on the Hub (under `danelcsb`):**
> [**model**](https://huggingface.co/danelcsb/localagent-tiny-30m-byte) ·
> [**live WebGPU demo**](https://huggingface.co/spaces/danelcsb/localagent-webgpu) ·
> [**dataset**](https://huggingface.co/datasets/danelcsb/localagent-dispatch-data)

## TL;DR — generable tool dispatch (latest)

A **28M, from-scratch, byte-level** agent that maps a natural request to a tool call over a
**50-tool** surface, via a **generable** pipeline (no fixed-N classifier — adding a tool is one row,
no retraining):

    request → route head (5-way modality gate) → dense two-tower selector (scores ANY tool by its
              description embedding) → pointer-copy arguments → tool(args)

Trained on a **corrected, paraphrase-rich + referent-conditioned** dataset. Held-out results
(disjoint phrasings & slot values):

| metric | score |
|---|---|
| free-form OOD call-name (45 hand-written) | **53%** (top-1 56%) |
| paraphrase-eval selection (100) | **63%** |
| referent-conditioned (URL vs file vs app) selection (46) | **72%** |
| multi-turn next-tool selection (27) | **74%** |

**Key findings** (full write-up in [`docs/DISPATCH_RESULTS.md`](docs/DISPATCH_RESULTS.md)): selection
needs a *trained* component (a fixed-N classifier carries it but doesn't scale; free-generation at
28M is ~0%); **arg values must be copied, not generated**; training has a **sweet spot** (~3.2k
steps — longer *degrades* OOD); and the SOTA "RL→on-policy-distillation" / specialist-distillation
moves **don't transfer** to a 28M model (capacity-bound). Deploys via
`Agent.from_checkpoint(ckpt, tools)`; exports to ONNX/JSON for the browser (parity-checked 100%).

---

> Below is the broader from-scratch pipeline (ToolCaller, the 3 size tiers, training/export/flywheel)
> that this dispatch work sits on top of.

## Reliable tool calling on *your* tools — no training required

Give `ToolCaller` any JSON-schema tools (multi-argument, real APIs) and it returns a
**schema-valid, grounded** call — or abstains. Selection scales to thousands of tools via
retrieval; arguments are filled by a schema-guided constrained decoder, so **the output is never
malformed**. No model download, no fine-tuning.

```python
from localagent import ToolCaller
from localagent.data.schema import ToolSpec

tools = [ToolSpec("move_file", "move or rename a file", {
    "type": "object",
    "properties": {"source": {"type": "string", "format": "path"},
                   "dest":   {"type": "string", "format": "path"}},
    "required": ["source", "dest"]})]

caller = ToolCaller(tools)
caller.call("Move src/app.py to backup/app.py.")   # ToolCall(move_file, {source:'src/app.py', dest:'backup/app.py'})
caller.call("What's the weather?")                  # None  (abstains)
```

**Benchmark** (`scripts/toolcall_eval.py` — 18 realistic multi-arg tools, paraphrased held-out
queries, disjoint slot values):

| | full-call | tool@1 | abstention |
|---|---|---|---|
| 18 tools | **72%** | 86% | — |
| 18 tools (`--min-score 0.12`) | 69% | 83% | **75%** |
| + **1,000 distractor tools** | 58% | 72% | — |

Full-call = correct tool *and* every argument grounded exactly. See `docs/TOOL_CALLING.md`.
Remaining misses are free-text multi-word args (the pointer-head frontier).

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

### Detailed results & development history
The full experimental narrative — the ultra-tiny ~1M flywheel, the 21-tool scaling study, the
28M training, parallel two-call turns, and the distillation negative — lives in
[`docs/HISTORY.md`](docs/HISTORY.md). The current generable-dispatch results:
[`docs/DISPATCH_RESULTS.md`](docs/DISPATCH_RESULTS.md).


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
| **`ToolCaller`** — schema-guided constrained decoding on any JSON-schema tools (multi-arg, retrieval, abstention), no training | ✅ implemented (`agent/caller.py`, `agent/schema_decode.py`) |
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
| Export to Hugging Face Hub (config + safetensors + heads + model card) | ✅ implemented (`scripts/push_to_hf.py`) — published: [`danelcsb/localagent-tiny-30m-byte`](https://huggingface.co/danelcsb/localagent-tiny-30m-byte) |
| Export GGUF/ONNX/ExecuTorch | 🚧 stubs (Phase 9) |

Full plan: [`docs/ROADMAP.md`](docs/ROADMAP.md). Stubs raise `NotImplementedError` with a
`TODO(phase-N)` pointing at the roadmap, so nothing silently pretends to work.

## License
MIT

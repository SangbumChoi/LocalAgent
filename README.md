# LocalAgent

A **< 100M-parameter** LLM, **pretrained from scratch**, that works as a local **agent** —
it does **tool calling** and **text generation**. Built as a **minimal, hackable, pure-PyTorch**
ML pipeline (nanochat in spirit), with **training + evaluation**, that runs on **CPU / GPU / NPU**
and exports to **PyTorch / GGUF / ONNX / ExecuTorch**. A **data flywheel** lets it improve from
the conversations it has while running locally.

> **🤗 Pretrained checkpoint:** the 28M byte-level agent is on the Hub —
> **[SangbumChoi/localagent-tiny-30m-byte](https://huggingface.co/SangbumChoi/localagent-tiny-30m-byte)**
> (config + safetensors + tool/pointer heads + model card). **71.3% single-turn** held-out **and
> 74% multi-turn** step-accuracy — strong on both. Loads in pure PyTorch, no `transformers`.

> **🚀 Live WebGPU demo:** run the agent **client-side in your browser** (no server) —
> **[danelcsb/localagent-webgpu](https://huggingface.co/spaces/danelcsb/localagent-webgpu)**.
> The transformer runs on `onnxruntime-web` (WebGPU, WASM fallback); the tool head, pointer-head
> grounding, and the `plan_rollout` are ported to JS. Source in [`spaces/localagent-webgpu/`](./spaces/localagent-webgpu/).

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

**Failure-driven flywheel (`scripts/analyze_loop.py`).** A real data flywheel: each round it tests,
**analyzes per-tool accuracy, and oversamples the weakest tools** (`weight = 1 + k·(1−acc)`) in the
next round's training data. Looped 5×, held-out accuracy climbs monotonically **45→50→57→61→62%**;
weak tools recover (`run_command` 0→100%, `read_file` 0→80%), while a few same-shaped quoted-arg
tools (`git_commit`) stay at 0% — oversampling can't fix what the 22-way 1M selector can't separate.
See `runs/analyze/analyze.png`.

**More data helps too (data-starved vs capacity-limited).** Expanding the dataset ~3× (slot pools
24→48-61 values/tool + more phrasings → 8,000+ unique samples, 0 grounding misses) lifts the *same*
1M model on the same failure-driven loop from **62% → 71%** held-out. So the two levers buy
different things: **data** 62→71% (the data-starved tools), **model size** 1M→28M round-1 45→64%
(the same-shaped tools the small head can't separate). See
`runs/analyze_ultra-tiny-1m/dataset_compare.png`.

### Pretrained 28M agent — strong on both axes 🤗
The deployable checkpoint is the **`tiny-30m-byte`** model (byte-level, `d_model` 512 × 10 layers,
GQA, pretrained from scratch). It is on the Hub:
**[SangbumChoi/localagent-tiny-30m-byte](https://huggingface.co/SangbumChoi/localagent-tiny-30m-byte)**.

Earlier configs forced a trade-off — the single-turn-only deploy hit ~83% single-turn but only ~18%
multi-turn, while joint multi-turn tuning lifted episodes but dragged single-turn down each round.
Two training changes close the gap: **gradient accumulation** (effective batch 32 inside an 11.8 GB
CPU footprint, so the small-batch noise that was sinking single-turn is gone) and a **down-weighted
multi-turn head** (`mt_weight=0.3`, so episode contexts stop pulling the shared tool head off the
single-turn distribution). The result is the first config that is **strong on both at once**:

| axis | this 28M | single-turn-only deploy |
|---|---|---|
| single-turn held-out (21 tools, disjoint slot values) | **71.3%** | ~83% |
| multi-turn step-accuracy (coding episodes) | **74%** | ~18% |

Reproduce: `python scripts/analyze_loop.py --model configs/model/tiny-30m-byte.yaml --rounds 4
--batch 16 --accum 2 --mt-weight 0.3 --episodes 120`. Load it (pure PyTorch, no `transformers`):

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from localagent.model import LocalAgentLM, ModelConfig

repo = "SangbumChoi/localagent-tiny-30m-byte"
cfg_d = json.load(open(hf_hub_download(repo, "config.json")))
cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
model = LocalAgentLM(cfg)
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
model.eval()
```

**Parallel two-call turns + 4× rebalanced dataset.** Real usage is "do X *and* Y" (two tool calls),
not a calc-dominated single-call stream. The dataset adds a **parallel** category (one turn → two
calls, e.g. *"Compose an email to Judy and search for how tall is Everest"* → `send_email` +
`web_search`), is **4× larger** (10,000 unique samples), and is **rebalanced** (`REALISTIC_WEIGHTS`:
parallel ~70%, calc ~6% — was ~70%). The decoder handles two-call turns by splitting on "and" and
grounding each conjunct (tool head + pointer head); eval matches the whole call *set*. On the
failure-driven loop the ~1M model learns parallel calls **0 → 38%** (both calls must be exactly
right) while overall climbs **32 → 63%**. See `runs/analyze_ultra-tiny-1m/parallel_result.png`.

**Scaling the model fixes selection (1M → ~28M byte tier, `configs/model/tiny-30m-byte.yaml`).**
On the *same* 21-tool data, round-1 held-out accuracy jumps **45% → 64%**; tool selection improves
across the board (`open_url` +83, `run_command` +62, `web_search` +62, `define` +53) and
`git_commit` goes 0%→17% — confirming the bottleneck was selection *capacity*, not data. (28M is
~6.5 s/step on a 4-core CPU, so this is a 2-round probe, not the full 5-round loop.) See
`runs/analyze_tiny-30m-byte/size_compare.png`.

**Distillation (`train/distill.py`, `scripts/distill_demo.py`) — implemented, honest negative.**
Offline logit-KD from the 28M teacher (held-out top-1 87%) into the 1M student. The distilled
student (67%) does **not** beat the SFT-only student (67%), and NLL gets worse — because on
*deterministic* templated tool-call targets, hard-label SFT is already near-optimal, so there's
little soft "dark knowledge" to transfer and temperature-softening hurts sharp next-byte
prediction. Distillation is the right tool for *ambiguous/open-ended* targets or for distilling the
capacity-limited skill directly (tool-head logits) — not deterministic copies. See
`runs/distill/distill.png`. (Both forward- and reverse-KL are implemented.)

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
| Export to Hugging Face Hub (config + safetensors + heads + model card) | ✅ implemented (`scripts/push_to_hf.py`) — published: [`SangbumChoi/localagent-tiny-30m-byte`](https://huggingface.co/SangbumChoi/localagent-tiny-30m-byte) |
| Export GGUF/ONNX/ExecuTorch | 🚧 stubs (Phase 9) |

Full plan: [`docs/ROADMAP.md`](docs/ROADMAP.md). Stubs raise `NotImplementedError` with a
`TODO(phase-N)` pointing at the roadmap, so nothing silently pretends to work.

## License
MIT

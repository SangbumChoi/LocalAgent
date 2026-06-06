# LocalAgent — Architecture

A minimal, hackable, **pure-PyTorch** pipeline for a **< 100M-parameter** LLM that acts as a
local **agent** (tool calling + text generation). Design choices are justified in
[`RESEARCH.md`](./RESEARCH.md); build order is in [`ROADMAP.md`](./ROADMAP.md).

---

## 1. System at a glance

```
                         ┌───────────────────────────────────────────────┐
                         │                  PIPELINE                      │
                         │  (localagent.pipeline.flow — orchestrates)     │
                         └───────────────────────────────────────────────┘
                                            │
   DATA                  TRAIN                          EVAL              SERVE
 ┌────────────┐   ┌──────────────────────┐      ┌──────────────┐   ┌──────────────┐
 │ pretrain   │   │ pretrain (from scratch)│      │ tool_eval    │   │ agent.runtime│
 │ corpus     │──▶│ sft (chat + tools)    │─────▶│  (BFCL-AST)  │──▶│ + tools      │
 │ agent_synth│   │ distill (rev-KL / OPD)│      │ text_eval    │   │ + memory     │
 │ flywheel   │   │ rl (GRPO, optional)   │      │ multi-turn   │   │ demos/ (UI)  │
 └────────────┘   └──────────────────────┘      └──────────────┘   └──────┬───────┘
       ▲                                                                   │
       │                                                                   ▼
       │                       ┌──────────────────────────────┐    ┌──────────────┐
       └───────────────────────│      DATA FLYWHEEL            │◀───│ conversation │
         mined + verified      │ ingest → mine → verify →      │    │   store      │
         training samples      │ schedule retrain → eval →     │    │ (+feedback)  │
                               │ redeploy                      │    └──────────────┘
                               └──────────────────────────────┘
                                            │
                      EXPORT (inference/export): PyTorch ▸ GGUF ▸ ONNX ▸ ExecuTorch
                      runs on CPU / GPU / NPU
```

Everything is a small Python module under `src/localagent/`, driven by YAML configs in
`configs/` and a single CLI (`localagent ...`). No training framework; just PyTorch + a BPE
tokenizer.

---

## 2. The model (`localagent.model`)

A compact Llama-style decoder, sized to stay **< 100M params**, favoring **depth over width**
and **GQA** (per SmolLM2).

- **Blocks:** pre-norm RMSNorm → GQA self-attention (RoPE positions) → SwiGLU MLP, residual.
- **Tied** input/output embeddings (the embedding table dominates param count at this scale).
- **KV cache** for generation; sliding-window optional.
- **Three tiers** (`configs/model/`) — each with a distinct job, not just "the same model,
  smaller" (see [`ARCHITECTURE_IDEAS.md`](./ARCHITECTURE_IDEAS.md)):

  | config | d_model | blocks×loops | heads / kv | ffn | vocab | ~params | role |
  |---|---|---|---|---|---|---|---|
  | `ultra-tiny-1m` | 192 | 2×6 (depth 12) | 6 / 2 | 640 | byte 256 | ~1.0M | tool router |
  | `tiny-30m`  | 384 | 12×1 | 6 / 2  | 1024 | 32k | ~31M | agent |
  | `small-90m` | 640 | 16×1 | 10 / 2 | 1728 | 32k | ~89M | capable agent |

  The ultra-tiny tier adds three structural levers — **byte vocab**, **factorized embeddings**
  (`embed_dim`), and **depth-recurrence** (`n_loops`, shared-weight passes) — that make ~1M
  feasible. Param budget is asserted at construction so configs can't silently exceed 100M.

**Tokenizer** (`model/tokenizer.py`): a byte-level BPE (trained with the `tokenizers` lib) with
**agent special tokens** so tool use is in-vocabulary:
`<|system|> <|user|> <|assistant|> <|tool|>`, `<tool_call>` / `</tool_call>`,
`<tool_response>` / `</tool_response>`, and `<|eot|>`.

---

## 3. Agent format & runtime (`localagent.agent`)

### Wire format (ChatML-ish, tool-native)
```
<|system|>
You are LocalAgent. Tools:
{json schema of available tools}
<|user|>
What's the weather in Paris?
<|assistant|>
<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call><|eot|>
<|tool|>
<tool_response>{"temp_c": 19, "cond": "cloudy"}</tool_response>
<|assistant|>
It's 19°C and cloudy in Paris.<|eot|>
```
The model decides between **emitting text** and **emitting a `<tool_call>`** (or abstaining — the
irrelevance case from Hammer/BFCL).

### Runtime loop (`agent/runtime.py`)
```
build prompt(system+tools+history+memory) → generate →
  parser.extract_tool_calls():
    if tool_call  → tools.dispatch() → append <tool_response> → loop
    else (text)   → return assistant text
```
- `agent/tools.py` — a **tool registry** (`@tool` decorator → name, JSON schema, callable),
  with a sandbox boundary and a few built-ins (calc, web stub, memory tools).
- `agent/parser.py` — robust extraction of tool calls into an **AST** (also reused by the
  evaluator) + tolerant JSON repair.
- `agent/memory.py` — **two-tier MemGPT-style memory**: `core` (always in context) + `archival`
  (out-of-context, searchable), exposed to the model as tools (`memory_append`,
  `memory_search`, `memory_replace`) with a paging/consolidation policy.

---

## 4. Data (`localagent.data`)

- `schema.py` — canonical dataclasses: `Message`, `ToolSpec`, `ToolCall`, `Conversation`,
  `Sample`. One JSONL line = one `Conversation`. This is the single interchange format across
  synth, flywheel, SFT, and eval (APIGen/xLAM-style).
- `pretrain_corpus.py` — streaming/packing of a quality-filtered text corpus into token shards
  (data-centric, SmolLM2-style). Toy default downloads a small public sample.
- `agent_synth.py` — the **synthetic agent-data generator** (ToolACE + APIGen + Hammer):
  1. sample tools from an API pool, 2. multi-agent dialog synthesis at a **target complexity**,
  3. inject **irrelevance negatives**, 4. **dual verification** (rule: schema/AST valid; model:
  semantic), 5. emit verified `Conversation`s. Pluggable "teacher" backend (any OpenAI-style
  endpoint or a local model).
- `flywheel.py` — ingest logged conversations + feedback from the store, **mine** good
  trajectories, **verify** them through the same dual checker, and append to the SFT/distill
  pool. Closes the loop in §7.

## 5. Training (`localagent.train`)

All stages share `train/device.py` (one **device abstraction**: CUDA / MPS / CPU / XPU / other
accelerators, autocast + dtype policy) so the *same* loop runs on GPU, CPU, or NPU-ish backends.

| stage | file | objective |
|---|---|---|
| Pretrain | `pretrain.py` | next-token CE on packed corpus shards; AdamW/Muon, cosine LR, grad accum, checkpointing |
| SFT | `sft.py` | next-token CE on `Conversation`s, **loss-masked** to assistant + tool-call spans; **function masking** (Hammer) |
| Distill | `distill.py` | **off-policy seq-KD** (default) and **on-policy reverse-KL** (MiniLLM/OPD) from a teacher; trajectory-level |
| RL | `rl.py` | optional **GRPO** with a tool-correctness + task-success reward |

Checkpoints are plain `state_dict` + config, resumable.

## 6. Evaluation (`localagent.eval`)

- `tool_eval.py` — **AST-based** BFCL-style scoring: parse predicted vs reference calls, compare
  trees; metrics for **single / parallel / multiple / multi-turn** and an **irrelevance**
  (correct-abstention) score.
- `text_eval.py` — perplexity + a few task accuracies (small ARC/GSM-style sets) for the
  text-generation side.
- `harness.py` — runs a config'd suite, writes a JSON report; also drives a **multi-turn**
  simulated-user + tool-sandbox eval (tau-bench-lite).
- Exported models are checked for **parity** vs the PyTorch reference here.

## 7. Conversation store + data flywheel (`localagent.agent.memory` + `localagent.data.flywheel`)

Every served conversation is persisted (SQLite to start) with optional **feedback signals**
(Airbnb AITL): pairwise preference, adoption decision + rationale, knowledge-relevance,
missing-knowledge. The flywheel job:

```
log conversations ─▶ mine candidates ─▶ dual-verify ─▶ append to train pool
        ▲                                                        │
        └──────────── redeploy ◀── eval ◀── distill/SFT ◀────────┘
```
This is what makes the system improve from real local usage instead of staying static.

## 8. Inference & export (`localagent.inference`)

- `generate.py` — KV-cached sampling (greedy / temp / top-p) used by the agent runtime and eval.
- `export/` — one converter per target: `to_gguf.py`, `to_onnx.py`, `to_executorch.py`, plus a
  shared **Q4_0-style quantizer** and a **parity** check. Goal: the same trained weights run on
  CPU/GPU/NPU through whichever runtime fits the device.

## 9. Demos (`demos/`)

- `chat_cli.py` — terminal agent chat (tool calls + memory visible).
- `web/app.py` — a small web UI (Gradio) that **visualizes** the agent loop: the live
  tool-call/tool-response trace, token stream, and memory state — the "visualizing the demos"
  goal.

## 10. Orchestration (`localagent.pipeline.flow` + `scripts/speedrun.sh`)

`flow.py` wires stages into a runnable DAG (`pretrain → sft → distill → eval → export`) with
artifact tracking; `speedrun.sh` runs the entire thing at toy scale on one machine (CPU-OK), the
nanochat-style "one command, end to end" entry point.

---

## Conventions
- **Configs over flags:** every stage takes a YAML in `configs/`; the CLI just selects one.
- **One interchange format:** the `Conversation` schema flows through synth → flywheel → train →
  eval. Add a field there, everything downstream sees it.
- **Stubs are honest:** unimplemented functions raise `NotImplementedError` with a `TODO(phase-N)`
  pointing at the roadmap. The README status table tracks what's real.

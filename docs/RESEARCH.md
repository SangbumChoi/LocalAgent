# Research: prior art and the good parts we steal

> Goal recap: a **< 100M-parameter** LLM, **pretrained from scratch**, that works as an
> **agent** (tool calling + free-form text generation), built as a **minimal, hackable,
> pure-PyTorch ML pipeline** (nanochat-style) with **training + evaluation**, running on
> **CPU / GPU / NPU**, exportable to **PyTorch / GGUF / ONNX / ExecuTorch**, and improved over
> time by a **data flywheel** that learns from stored conversations.

This document surveys the projects worth learning from and records, for each, **only the parts
we adopt** and why. The design in [`ARCHITECTURE.md`](./ARCHITECTURE.md) is the synthesis of
these decisions.

---

## 1. Full-stack minimal trainers

### nanochat — Karpathy ([repo](https://github.com/karpathy/nanochat))
A single, dependency-light codebase covering the *whole* lifecycle: tokenizer → pretrain →
midtrain → SFT → (optional) RL with GRPO → eval → a chat web UI, runnable end-to-end via one
`speedrun.sh` for ~$100 / ~4h on one node.

**We adopt:**
- The **single cohesive codebase** philosophy — no giant config monsters, no framework. Every
  stage is a readable script you can fork.
- The **stage ordering**: tokenizer → pretrain → mid/SFT → (RL) → eval → serve.
- A **`speedrun.sh`** that runs the whole pipeline start-to-finish at toy scale.
- A **built-in chat UI** as the demo surface.
- **GRPO** as the optional RL stage (we apply it to tool-use correctness, not just math).

**We drop:** the assumption of an 8×H100 node. Our default target is a single small GPU *or*
CPU, at <100M params.

### nanoGPT / modded-nanoGPT — Karpathy & community
Minimal GPT model + training loop; modded-nanoGPT is the speedrun playground (Muon optimizer,
architectural tweaks).

**We adopt:** the compact `model.py` + `train.py` skeleton, and Muon/AdamW as a tunable
optimizer choice for pretraining.

---

## 2. Tiny-model architecture & data

### SmolLM2 (135M/360M/1.7B) — HuggingFace ([paper](https://arxiv.org/abs/2502.02737))
Data-centric training of genuinely small, on-device models.

**We adopt:**
- **Grouped-Query Attention (GQA)** for cheap inference at small scale.
- **Depth over width** (more layers, narrower `d_model`) for the <100M budget.
- **Data-centric curation** over raw web scrapes — quality-filtered, structured corpora.
- The 135M point as a **reference size/recipe sanity check** for our from-scratch model.

---

## 3. Agent / tool-calling data generation

### APIGen / xLAM — Salesforce ([xLAM-function-calling-60k])
A verifiable, multi-stage synthetic data pipeline for function-calling, with format checking,
**actual execution checking**, and semantic verification; ships a 60k function-calling dataset
with a clean JSON schema.

**We adopt:** the **multi-stage verification** idea (format → execution → semantic) and the
**function-calling JSON schema** as our on-disk format for agent samples.

### ToolACE ([paper](https://arxiv.org/abs/2409.00920))
An agentic **self-evolution** synthesis loop over a large API pool, a **complexity evaluator**
that targets samples at the model's level, and a **dual-layer (rule + model) verifier**.

**We adopt:** the **multi-agent synthesis loop**, the **complexity-targeting** of generated
samples, and the **rule-based + model-based dual verification** of every sample.

### Hammer ([repo](https://github.com/MadeAgents/Hammer))
On-device function-calling robustness via **function masking** and an
**irrelevance-augmented** dataset (examples where *no* tool should be called).

**We adopt:** **irrelevance / "no tool needed" negatives** as a first-class part of the dataset,
and **function masking** during training so the model generalizes across tool name/arg surface
forms instead of memorizing them.

---

## 4. Distillation (the "distillation part")

### MiniLLM ([paper](https://arxiv.org/abs/2306.08543)) & On-Policy Distillation (Thinking Machines, MiniPLM)
Reverse-KL instead of forward-KL for generative KD; **on-policy** distillation where the
*student* generates trajectories and the *teacher* scores them (fixes exposure bias).

**We adopt:** **reverse-KL** as the KD objective and an **on-policy loop** (student samples →
teacher logits/scores → update) as the advanced distillation mode. A simpler **off-policy
sequence-KD** mode (teacher generates, student imitates) is the default starter.

### Distilling LLM Agents into Small Models ([2505.17612])
Distill **full agentic trajectories** (with tool invocations) rather than just CoT; tricks like
**first-thought prefix** and **self-consistent action selection** let 0.5–3B students match the
next tier up.

**We adopt:** **trajectory-level distillation** (the teacher's whole tool-use rollout, including
reasoning + tool calls + tool responses, is the training signal) and **first-thought prefix**
prompting when generating teacher data.

---

## 5. Memory & the data flywheel

### MemGPT / Letta ([docs](https://docs.letta.com/concepts/letta/))
LLM-as-OS: a **memory hierarchy** (in-context "core" memory vs out-of-context archival), the
model **self-edits** memory via tools, and a pager moves data in/out of the context window.

**We adopt:** a **two-tier memory** (core/in-context + archival/out-of-context), exposed to the
model **as tools** (`memory_append`, `memory_search`, …) so memory management is itself agent
behavior, plus a simple **paging/consolidation** policy.

### Agent-in-the-Loop data flywheel — Airbnb ([2510.06674])
A production flywheel that captures four annotation types — **pairwise preferences**, **adoption
decisions + rationale**, **knowledge-relevance checks**, **missing-knowledge flags** — and feeds
them back into model updates, cutting retrain cycles from months to weeks.

**We adopt:** the **flywheel schema** (every stored conversation can carry these feedback
signals) and the **loop**: log conversations → mine/verify into new training samples → schedule
a retrain/distill → evaluate → redeploy.

---

## 6. Evaluation

### BFCL — Berkeley Function Calling Leaderboard ([proceedings](https://proceedings.mlr.press/v267/patil25a.html))
The standard tool-use benchmark: **AST-based** evaluation of single/parallel/multiple calls,
multi-turn (v3+), and **irrelevance detection** (should the model abstain from calling?).

**We adopt:** an **AST-based tool-call evaluator** (compare parsed call trees, not strings),
plus **multi-turn** and **irrelevance** splits in our eval harness.

### tau-bench / ToolBench
Multi-turn tool use in realistic customer-service / real-API settings.

**We adopt:** a small **multi-turn task harness** with a simulated user + tool sandbox for
end-to-end agent eval beyond static call matching.

### Tool/function-calling benchmark landscape — survey ([BFCL_v4 survey](https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information))
A survey that maps the tool-calling eval space along three axes — **task complexity** (single →
parallel/sequential → agentic planning), **interaction model** (stateless single-turn → stateful
multi-turn), and **evaluation method** (AST match → execution → outcome → LLM-judge) — and names the
core tension: *more realistic benchmarks force less reliable scoring* (open-ended state pushes you
toward LLM-as-judge with its position/verbosity biases). It documents a **mechanics-vs-strategy**
split — e.g. GPT-4o ~84% on stateless BFCL but ~7% on a stateful variant — i.e. grounded call
*mechanics* and long-horizon *strategy* are distinct competencies. Beyond BFCL/ToolBench it catalogs
**API-Bank** (decomposes tool use into planning / retrieving / calling), **ToolSandbox** (stateful
multi-turn scored by a Milestone DAG), AgentBench, and MCP-AgentBench.

**We adopt:** the **AST/execution end** of that ladder (score grounded call trees against a
deterministic oracle, never an LLM judge); **API-Bank's planning/retrieving/calling decomposition**,
which maps cleanly onto our own split of *tool selection* (the head) vs *argument grounding*
(pointer/heuristics) vs *abstention*; and the **mechanics-vs-strategy** finding as direct support for
scoping the 28M model to mechanics and leaving strategy to the orchestrator (see the ALE note below).

### tool-calling-benchmark — Veerman ([repo](https://github.com/MikeVeerman/tool-calling-benchmark))
Exactly our regime: **small open-weight models (0.5–3.8B), CPU-only**, scored on *judgment*. 12
single-turn prompts mix actionable calls, **restraint** cases (where calling *any* tool is wrong,
e.g. "what tools do you have?"), and **misleading** prompts that bait keyword-triggered calls.
Scoring is a weighted **Agent Score = Action·0.4 + Restraint·0.3 + Wrong-Tool-Avoidance·0.3**,
aggregated by **majority vote over 20 runs** (3 runs proved unreliable). Top small model: qwen3:1.7b
at 0.960.

**We adopt:** **restraint/abstention as a first-class *scored* axis** (our irrelevance negatives
already train it — this says to weight and report it alongside action, not bury it); an explicit
**wrong-tool penalty** for same-shaped distractor tools (our known 22-way selector confusion); and
**multi-run majority-vote** for stable tiny-model eval on CPU, where single-run accuracy is noisy.

### Agents' Last Exam (ALE) ([abs](https://arxiv.org/abs/2606.05405) · [site](https://agents-last-exam.org)) — *contrast, not a target*
The frontier end of the spectrum: **1,000+ long-horizon, economically-valuable real-world tasks**
with **verifiable outcomes**, spanning 55 subfields across 13 industry clusters (U.S. occupational
taxonomy), built with 250+ industry experts. It is deliberately unsaturated — mainstream frontier
configs average just **~2.6% full pass** on the hardest tier. This is autonomous *professional
workflow* evaluation, the opposite pole from a <100M single-/parallel-call tool router.

**We don't target it** — a 28M on-device model is a **controller**, not an autonomous long-horizon
agent; sustained multi-step workflows are the *orchestrator's* job, with the tiny model supplying
grounded tool calls underneath. What we **do** take is the **verifiable-outcome** discipline (score
against an executable/AST oracle, never an LLM judge) and ALE as an explicit **scope boundary**: it
keeps us honest that our held-out tool-call accuracy measures the router, not end-to-end autonomy.

---

## 7. On-device inference / export ("runs on CPU/GPU/NPU")

| Runtime | Strength | Our use |
|---|---|---|
| **PyTorch** (eager) | dev loop, ground truth | reference + correctness oracle |
| **GGUF / llama.cpp** | CPU + Apple Silicon + Intel/AMD GPU/NPU (via OpenVINO upstream) | primary "runs everywhere" path |
| **ONNX Runtime** | one graph, Execution Providers for CPU/GPU/NPU (incl. AMD Ryzen AI) | unified backend across desktop/mobile/cloud |
| **ExecuTorch** | ~50KB runtime, native PyTorch AOT export | smallest mobile / microcontroller / NPU footprint |

**We adopt:** a single **`export/`** module with one converter per target, **Q4_0-style
quantization** as the default on-device format, and a **parity test** that checks each exported
model against the PyTorch reference on a fixed prompt set.

---

## Summary table — what each project contributes

| Project | One thing we take |
|---|---|
| nanochat | single-codebase full pipeline + `speedrun.sh` + chat UI + GRPO |
| nanoGPT | compact model/train skeleton + optimizer choice |
| SmolLM2 | GQA + depth-over-width + data-centric curation @ <100M |
| APIGen/xLAM | multi-stage verification + function-calling JSON schema |
| ToolACE | multi-agent synthesis + complexity targeting + dual verify |
| Hammer | irrelevance negatives + function masking |
| MiniLLM / OPD | reverse-KL + on-policy distillation |
| Agent-distill (2505.17612) | trajectory-level distillation + first-thought prefix |
| MemGPT/Letta | two-tier self-editing memory exposed as tools |
| Airbnb AITL | data-flywheel feedback schema + retrain loop |
| BFCL / tau-bench | AST tool eval + multi-turn + irrelevance |
| Tool-calling survey (BFCL_v4) | AST/execution-end scoring + planning/retrieving/calling split + mechanics-vs-strategy |
| tool-calling-benchmark (Veerman) | restraint as a scored axis + wrong-tool penalty + multi-run majority vote |
| Agents' Last Exam | verifiable-outcome discipline + a scope boundary (contrast, not a target) |
| llama.cpp/ONNX/ExecuTorch | four export targets + Q4 + parity test |

---

## See also — computer-use & tool-calling agents

For the recent **computer-use / multi-agent-browser / tool-calling** wave (UI-TARS, OSWorld-G/Jedi,
WebVoyager, SeeAct, Mind2Web, WebArena, Gorilla, ToolLLM, xLAM/APIGen, Hammer, ToolACE, ToolAlpaca,
CodeAct, OctoTools, AutoGen, CoAct-1) — with each paper's good part, a two-persona **scale vs
structure** debate, and the verdict for a sub-100M on-device agent — see
[`COMPUTER_USE_DEBATE.md`](./COMPUTER_USE_DEBATE.md). That survey is what motivated
`ToolCaller.plan()` (planner over grounded calls).

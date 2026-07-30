# LocalAgent

A **< 100M-parameter** LLM, **pretrained from scratch**, that works as a local **agent** —
it does **tool calling** and **text generation**. Built as a **minimal, hackable, pure-PyTorch**
ML pipeline (nanochat in spirit), with **training + evaluation** across CUDA, MPS, XPU, and CPU
for training and reference inference. PyTorch and ONNX/WebGPU export work today; GGUF and
ExecuTorch remain planned. The deterministic research flywheel is implemented, while production
ingestion of served conversations and feedback is not.

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
`Agent.from_checkpoint(ckpt, tools)`; exports to ONNX/JSON for the browser. Every emitted fp32/fp16
graph must pass output-specific numerical parity thresholds before the exporter atomically
publishes a bundle manifest; benchmark pages then verify the fetched graph bytes against it.

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
- **Portable training** — one device abstraction covers CUDA/MPS/XPU/CPU; ONNX/WebGPU export is
  implemented, while GGUF and ExecuTorch remain roadmap items.
- **Capacity experiments without budget fiction** — an opt-in Micro-MoE executes only selected
  experts in PyTorch and reports both all stored parameters and the nominal active token path;
  dense remains the WebGPU control until sparse export and browser measurements pass.
- **Research flywheel** — deterministic synthesize/train/evaluate/enrich loops are implemented;
  production storage, mining, and feedback ingestion remain roadmap items.

The design synthesizes the best ideas from prior work — see [`docs/RESEARCH.md`](docs/RESEARCH.md)
for what we take from nanochat, SmolLM2, APIGen/xLAM, ToolACE, Hammer, MiniLLM/on-policy-
distillation, MemGPT/Letta, the Airbnb data flywheel, and BFCL — but it isn't bound by them:
[`docs/ARCHITECTURE_IDEAS.md`](docs/ARCHITECTURE_IDEAS.md) lays out the original structural bets
(agent-as-controller, the vocabulary tax, factorized + depth-recurrent ultra-tiny models, a dual
tool/text head, grammar-constrained decoding). [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is
how it all fits together.
[`docs/TRAINING_SYSTEM.md`](docs/TRAINING_SYSTEM.md) is the current Kimi K2/K2.5/K3,
GLM/Grok/SOLAR/nanochat comparison and the concrete dataset → pretrain → midtrain → SFT → RL
recipe.
[`docs/SPARSE_EXPERTS_REAL_DATA.md`](docs/SPARSE_EXPERTS_REAL_DATA.md) defines the active-matched
43.86M sparse/17.30M dense experiment, pinned public xLAM/Mind2Web ingestion, and the quality,
router, and deployment promotion gates.
The runnable GPU path is
[`notebooks/localagent_pretraining_colab.ipynb`](notebooks/localagent_pretraining_colab.ipynb):
licensed corpus streaming, mixed-precision training, resumable checkpoints, and verified Google
Drive storage.

### WebGPU cached-decode deployment frontier

A separate random-weight systems benchmark now exports parity-gated fp16 prompt-prefill and
fixed-one-token decode graphs, keeps cache tensors in requested/reported `gpu-buffer` storage, and
rebinds them without JavaScript readback. On one 32 GB Apple M5 in Chrome 150 / ONNX Runtime Web
1.27.0, the median of three page-run p50 wall rates for the matched 10.5M hybrid were **159, 160,
143, and 128 tokens/s** at 128, 512, 1,024, and 1,536 input tokens. That clears the requested
100 tokens/s engineering reference across the tested contexts; matched 15.6M and 34.2M hybrids
did not.

These are deterministic **random-weight latency results**, not trained quality or agent results.
The cache uses append/concat rather than in-place or paged updates, separate prefill/decode
sessions may duplicate weights, and per-node placement/fallback is unknown. See the
[paper research plan](docs/paper/SLMW2026_RESEARCH_PLAN.md) and tracked
[results](docs/paper/results/).

An observed exploratory seed-2026 checkpoint-backed proxy used 10,551,291 pretraining loss tokens
per arm. The hybrid had better held-out CE, bits/byte, and top-1 token accuracy than the matched
attention arm; paired 10,000-resample document-bootstrap intervals exclude zero for all three
aggregate differences, conditional on that seed and its 240 validation documents.

A separate, prospectively designated clean confirmatory set covers seeds 2027–2029. The
attention-minus-hybrid BPB gaps were **+0.07084, +0.07353, and +0.07449**, for a mean of
**+0.07295**; hybrid was favored in all three seeds. A model-based Student-t interval with two
degrees of freedom is `[0.06825, 0.07765]`, but normality cannot be assessed with three seeds, and
the exact sign test is not conventionally significant (one-sided `p=0.125`, two-sided `p=0.25`).
The mean CE gap was +0.21045 and the hybrid's mean top-1 accuracy was 2.193 percentage points
higher. Confirmatory scorecards were CPU fp32. Training configs used `device: auto`, but the
runner did not persist the resolved training device; this host currently reports MPS unavailable
and resolves `auto` to CPU, which is not retrospective proof of where those runs trained. See the
[confirmatory summary](docs/paper/results/webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json) and
[raw scorecards](docs/paper/results/raw/pretrain-proxy-seeds2027-2029/).

Separately, the exact **seed-2026** pretrain-only hybrid checkpoint was exported and measured on
the same M5 browser stack. Its median-of-three p50 wall rates are **137.7, 128.2, 123.8, and
116.5 tokens/s**. It clears 100 tokens/s at every context in every run, but it does **not** clear
the full tail gate: median p95 TPOT is 10.31/9.61/10.21/10.21 ms, so only the 512-token condition
is at or below 10 ms, and no context passes the joint gate in every run. This is not confirmatory
seed-set latency or agent quality. The three-seed quality result makes the hybrid a provisional
choice for a bounded post-training pilot only; it does not complete the 34M five-TPP architecture
screen or the subsequent 20-TPP/downstream quality selection. See the
[seed-2026 trained-latency summary](docs/paper/results/m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json).

The provisional seed-2027 hybrid has now completed a bounded post-training pilot. Midtraining
improved held-out agent loss from **7.787 to 2.637** and token accuracy from **3.7% to 69.8%**,
while the much larger general holdout regressed slightly (loss **5.706 to 5.734**). SFT improved
held-out assistant loss from **2.732 to 1.815** and token accuracy from **67.3% to 73.1%**, but
teacher-forced all-assistant-token exactness was only **1/65**; this is not free-running
generation. Offline GRPO realized 12 optimizer updates, yet
all 53-row held-out metrics were unchanged; it is a negative RL result, not evidence of lift.
The exact stage lineage and runtime are in the
[pilot summary](docs/paper/results/webgpu-proxy-pilot-seed2027.summary.json).

The SFT checkpoint's parity-gated fp16 action graph was then measured in three M5/Chrome
WebGPU sessions under the internally prespecified pre-assistant-padding stress condition at fixed
512-token inputs. Median-of-run TTFA was **24.75 ms p50** and
**34.405 ms p95**, but the policy abstained on every case. The capability denominator was
**1/20** unique cases overall: **0/19** tool-required and **1/1** abstention. Across the 30 timing
repetitions in three sessions this becomes 90/1,800 exact and 0/1,710 tool-required, not 1,800
independent capability trials. A three-run local DOM loop likewise had **0/8** unique tasks
(0/720 timing opportunities) with exact actions,
executable schemas, final DOM successes, or closed-loop successes despite 33.3 ms pooled p50
closed-loop latency. The 100% schema figure in the action suite reflects valid abstention, not
useful tool output.

An exploratory offline diagnostic found good natural-prompt route/selector parity (17/20 routes,
17/19 tool selections), while the internally prespecified stress condition materializes real,
unmasked space tokens before the assistant marker and collapses every route to text at 128 tokens
or longer.
Native PyTorch, fp32/fp16 ONNX, and exported JSON heads agree, so export/precision mismatch is not
the cause. The fixed-512 browser result still fails the capability gate. A corrected runner now
keeps the 512-token compute while dispatching from the natural assistant-marker position and
restricting pointer scans to the natural prompt. A
[full-stack parity gate](docs/paper/results/sft-structured-export-parity-seed2027.summary.json)
gets exact native/fp32-ONNX/fp16-ONNX agreement on 20/20 reused-suite routes, tools, grounded
arguments, and normalized actions; the shared 16/20 offline exact score is diagnostic reuse, not
browser or independent capability evidence. Its
[browser protocol](docs/paper/results/webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json)
is preserved as a superseded historical proposal: no external timestamp or browser run was
collected under it. The current deployment contract instead uses lineage-bound cached
prefill/decode graphs for the two autoregressive controls.
Because the 20/8 suites and 65-row evaluation set
were inspected during diagnosis, those runs are reused-suite deployment-parity re-evaluations,
not untouched capability confirmation. Genuine fixed-512 capability would separately
require training and evaluation on the exact pre-marker materialization. This pilot is not
evidence of a useful WebGPU agent. See the
[action summary](docs/paper/results/m5-webgpu-sft-action-pilot-seed2027.summary.json) and
[result index](docs/paper/results/README.md). Final 35M/at-least-five-TPP, a trained cached
complete-action bundle and browser score, BrowserGym/open-web, and cross-device evidence remain
open.

### Shared-corpus 1M / 10M / 96M tiers

The current production experiment uses one frozen 16K-BPE tokenizer, one 2K packed paper corpus,
one validation split, and a 4K model context across all three tiers. That makes held-out bits per
byte and agent-call case IDs comparable without pretending that differently tokenized
perplexities are comparable. The longer model context is required by the frozen 50-tool
schema-conditioned scorecard: its longest paper-tokenizer prompt is 3,590 tokens, so the
prompt plus a 96-token decode allowance is 3,686
tokens.

| Tier | Exact LM params | Intended job | Structure | Current evidence |
|---|---:|---|---|---|
| `webgpu-1m-bpe-router` | 980,480 | tool router / planner | 32→128 factorized embeddings; shared `[conv, attn]` stack looped 3×; MQA | real pretrain → midtrain → base SFT + parent-anchor continuation complete; retained step 12 failed its bound development greedy gate, so confirmatory evaluation, RL, and candidate WebGPU promotion were withheld |
| `webgpu-10m-hybrid-4k` | 10,524,544 | latency-feasible autoregressive agent | four unique blocks in a `[conv, attn, conv, attn]` pattern; QK-Norm; MQA | architecture-matched to four historical 2K proxy seeds; new paper 5-TPP run has no score yet |
| `webgpu-96m-hybrid` | 95,320,448 | near-budget quality/feasibility arm | 18 layers, 12 short-conv + 6 attention; QK-Norm; MQA | architecture + full staged configs only; untrained and unbenchmarked |

The 10M and 96M hybrids have parameter-matched all-attention controls. Historical 2K 10M configs
and checkpoints remain unchanged, so their 116–138 p50 trained cached-decode evidence is not
silently relabeled as a 4K result. The 96M model's estimated 45.45 MiB Q4 weight packing is
arithmetic only: the current browser exporter runs FP16, where its weights are about 181.8 MiB per
graph, and it has no 100–300 tok/s claim. Its first local training run uses micro-batch one as an
explicit memory/speed preflight. The 96M midtrain, SFT, and RL configs likewise remain
memory-gated execution contracts, not evidence that those stages have run successfully.

The 1M paper lane is now an executed result rather than a config-only claim. Pretraining consumed
19,628,032 input tokens and reduced its fixed validation loss from 9.7041 to 5.7017. Midtraining
then consumed 21,680,586 input / 16,813,239 loss tokens and improved the same-draw aggregate
held-out loss from 5.0197 to 4.7589. The base SFT consumed 5,568 canonical assistant decisions and
improved its sealed 820-decision teacher-forced holdout from 6.4066 to 2.6378 loss and from 4.85%
to 63.30% token accuracy.

A separate MPS parent-anchor continuation then completed 372 updates: 348 exact parent-replay
updates plus 24 format pulses. It consumed 5,952 decisions, 20,932,540 input tokens, and 174,074
assistant-loss tokens while publishing 31 immutable archives. The frozen teacher-forced sweep
found only two retention-eligible archives and selected step 12
(`e1cf203368f6f19a8a46f0cd2a297bd61373e44fbf24155e3d68b5137db430c7`): loss 2.636792 and
11,687/18,460 correct assistant tokens. Step 372 lowered loss to 2.627632 but fell to
11,646/18,460 correct tokens, so it failed the zero-drop retention gate. Step 24 was eligible but
was never available as a fallback.

The retained step-12 result still did **not** become a useful free-running agent. Its independently
replayed internal BFCL-style development scorecard reached EOS on 170/820 generations (20.73%),
truncated 650/820, and scored zero on complete format, strict tool format, schema validity,
case-exact tool name, whole-call exactness, and structural abstention. The sealed
[promotion decision](data/provenance/paper/sft-candidate-parent-anchor-pulse-development-decision.json)
is `development_gate_failed`: confirmatory evaluation was not supplied, fallback was forbidden,
and RL plus candidate-specific WebGPU export were not authorized. The earlier parent checkpoint's
zero-reward RL preflight remains historical baseline evidence, not evidence about step 12. No
100–300 tok/s capability-qualified claim is made.

The recovery evidence is reproducible: the
[production receipt](data/provenance/paper/production/sft-paper-tier-1m-parent-anchor-pulse-pilot.json),
[candidate binding](data/provenance/paper/sft-candidate-parent-anchor-pulse-selected.json), and
development rejection are self-hashed. The earlier
[8,192-row format curriculum](data/provenance/paper/agent-sft-format-bootstrap-v1.json) and its
[512-update plan](data/provenance/paper/stage-budgets/sft-paper-tier-1m-format-bootstrap.json)
remain a separate rejected intervention. The current RL config still points to the failed parent
checkpoint; no candidate-bound RL run was fabricated.

Full-catalog midtraining is budgeted by actual input tokens: the 1M/10M/96M horizons are
599/1,610/2,909 updates, matching each tier's pretraining pilot. Using supervised loss tokens as
the mixture unit would hide the repeated 3.5K-token catalog prefix and inflate the 10M schedule to
1.386B input tokens, so that legacy setting is deliberately not carried into these tier configs.

The original `ultra-tiny-1m` byte model remains available and spends its budget more efficiently
than the BPE router, but it requires a byte-packed corpus. At ~1M parameters the goal is a
*controller*, not a chatbot: knowledge is offloaded to tools and retrieval so the parameters can
encode skills and control flow.

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
**[danelcsb/localagent-tiny-30m-byte](https://huggingface.co/danelcsb/localagent-tiny-30m-byte)**.

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

repo = "danelcsb/localagent-tiny-30m-byte"
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
# CPython 3.12 is the reproducible paper/test runtime. The core package still supports
# Python >=3.10, but the frozen Mind2Web ranker intentionally binds Unicode 15.0.
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,demo]"

# inspect a model config and check it's under the 100M budget
localagent model-info configs/model/tiny-30m.yaml

# run the toy end-to-end pipeline
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
| Opt-in sparse routed FFN with honest total/active counts, load balancing, and diagnostics | ✅ PyTorch research path; 🚧 sparse ONNX/WebGPU lowering and measured promotion |
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
| Tokenizer training | ✅ byte + trained BPE with atomic agent markers |
| Corpus filtering, provenance, deterministic split, memory-mapped packing | ✅ implemented (`scripts/prepare_corpus.py`) |
| Canonical agent-data export (single/multi-turn/irrelevance + schema verification) | ✅ implemented (`localagent synth ...`) |
| Pinned public xLAM/Mind2Web TRAIN ingestion + provenance/decontamination manifest | ✅ implemented (`scripts/ingest_public_agent_data.py`) |
| Pretrain / midtrain / SFT / Distill / exact-AST RL loops | ✅ core loops; config runners for pretrain, midtrain, SFT, and RL |
| Compatible pretrained-checkpoint depth growth | ✅ explicit layer map, self-hashed verification, fresh-optimizer `init_from`; not function-preserving |
| Agent data synthesis + research flywheel | ✅ deterministic synthesis/evaluation loop; 🚧 production feedback ingestion |
| Eval harness | ✅ AST, grounded, multi-turn, public real-use/router gates, realtime, and deterministic browser primitives; 🚧 unified frozen-suite runner/general text eval |
| Agent runtime + memory + demos | ✅ checkpoint-backed runtime and web demos; 🚧 two-tier memory and chat CLI |
| Export to Hugging Face Hub (config + safetensors + heads + model card) | ✅ implemented (`scripts/push_to_hf.py`) — published: [`danelcsb/localagent-tiny-30m-byte`](https://huggingface.co/danelcsb/localagent-tiny-30m-byte) |
| Export | ✅ ONNX/WebGPU + cache-bearing prefill/decode + Hugging Face; 🚧 GGUF, ExecuTorch, and real Q4 |

Full plan: [`docs/ROADMAP.md`](docs/ROADMAP.md). Remaining stubs raise `NotImplementedError`
with a `TODO(phase-N)` pointing at the roadmap, so nothing silently pretends to work.

## License
MIT

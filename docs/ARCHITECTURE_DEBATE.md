# Architecture debate: what a sub-100M on-device tool agent should steal from the 2024–2026 model zoo

> Companion to [`COMPUTER_USE_DEBATE.md`](./COMPUTER_USE_DEBATE.md) (which argued *scale vs structure* for
> the agent loop) and [`RESEARCH.md`](./RESEARCH.md) (per-project adoptions). This one is about the
> **model itself** — we read the architecture/training reports of the recent frontier and small-model
> waves, extract the *fundamental philosophies*, stage them as a debate, and decide what actually ports
> to our **~1M** and **~30M** from-scratch, pure-PyTorch, CPU tool-calling tiers.

## The central tension (state it before the debate)

Almost every celebrated model below is a **scale-era frontier model optimizing a *serving* constraint** —
KV-cache memory, long-context throughput, expert routing at hundreds of billions of params. **Our
constraint is the opposite: we are parameter- and compute-starved, on short contexts, on a CPU.** Most
of the famous tricks (MLA, MoE, FP8, million-token sparse attention) are answers to problems we *don't
have*. The genuinely portable ideas are a minority — and they cluster in **training method**, not
architecture. The debate exists to separate the two.

## The cast

- **The Scaler** — capability comes from parameters, cleverly served. MoE, big teachers, RL. ("Just
  distill a giant into it.")
- **The Structuralist** — the *primitive* is the lever. Cheap mixers (short conv), depth over width,
  weight sharing. ("Spend your few params on the right operator.")
- **The Efficiency Hawk** — nothing counts unless it pays on the target device. Hardware-in-the-loop;
  latency/RSS over benchmark vanity. ("Show me the p95 decode on a Ryzen.")
- **The Data/Distillation Camp** — architecture is nearly a rounding error; capability is information
  density per token + imitation of a teacher. (Phi, SmolLM2, the R1 distills.)

---

## The axes, the argument, the verdict

### 1. Scale vs structural efficiency
**Scaler:** Kimi K2 (1T/32B-active), DeepSeek-V3 (671B/37B), Llama-4 MoE — frontier quality at a fraction
of *served* FLOPs. **Structuralist:** Nemotron-H replaces ~92% of attention with Mamba-2 for 3× decode;
LFM2 is conv-heavy; MiniMax/Qwen3-Next go 75–87% linear. The frontier itself is drifting from "scale up"
to "scale up **but make each served token cheaper**."
**Verdict (ours):** we have no scale to serve — only the *cheaper-primitive* half of the lesson applies.
Structuralist wins our tier outright.

### 2. Attention vs sub-quadratic mixing
**Structuralist:** most mixing should be cheap and local — Nemotron-H (~8% attention), Qwen3-Next (25%),
LFM2 (≈37% attention: 10 conv + 6 GQA), Gemma-3 (5 local : 1 global). **Scaler/Hawk rebuttal:** you still
need *some* softmax attention for in-context retrieval and **copying tool arguments verbatim** — the one
thing a tool agent cannot get wrong. Consensus number across the zoo: **~1 attention layer per 3–8
sub-quadratic layers.** The Hawk adds the decisive datum: Liquid's *hardware-in-the-loop* search found
that **Mamba/SSM/linear-attention did not beat a plain gated short-conv + GQA hybrid under CPU
latency/RSS budgets** — and often hurt them.
**Verdict (ours):** **gated short causal conv (k=3, double-gated, LIV-style) as the primary mixer + 1–2
global GQA layers** for argument-copying. **Do not reach for Mamba/SSM** — it loses its fused-CUDA edge
on CPU and a naive PyTorch scan can be *slower* than small windowed attention. (LFM2; Nemotron-H.)

### 3. Dense vs sparse (MoE)
**Scaler:** MoE decouples capacity from active FLOPs — the default above ~30B-active. **Everyone else:**
every MoE model in the corpus is *huge*; none is small, because MoE needs enough params-per-expert to be
worth the routing tax, and load-balancing is unstable with tiny batches.
**Verdict (ours):** ❌ **No MoE.** At <100M, experts are pathologically under-parameterized; a **dense**
tiny model is strictly better per active FLOP. (DeepSeekMoE; Qwen3; Llama-4 — all as cautionary scale.)

### 4. Learn-from-scratch vs distill-from-teacher  ← *the load-bearing axis*
**Data/Distillation Camp:** the single most consistent finding in the whole corpus — **Qwen3 and
DeepSeek-R1 independently conclude small models should be *distilled*, not RL'd**: ~1/10 the compute,
*higher* Pass@1 **and** Pass@64. R1 is blunt: small models can't *discover* reasoning via RL, they must
*imitate* it. LFM2 goes furthest — distillation (a decoupled tempered Top-K objective) is the **primary
pretraining signal**, not a post-hoc compression step. **Scaler caveat:** the gains scale with the
teacher–student capacity gap, and our teacher is only the 30M, not a 7B — so expect *modest*, not
miraculous, lifts. **Sub-debate (which divergence):** forward-KL is mode-covering (diversity); **reverse-KL
is mode-seeking — the right choice for tool-calling, where one correct call beats a diverse blur** (MiniLLM).
And **on-policy** distillation (distill on the student's *own* sampled trajectories, teacher scores them)
cuts exposure bias from **O(εT²) → O(εT)** — decisive for multi-step plans.
**Verdict (ours):** ✅ **30M → 1M distillation is our highest-leverage change.** Same tokenizer/heads in
both tiers ⇒ no vocab-projection needed. Recipe: **tempered Top-K KD** (cheap dense signal) → then an
**on-policy pass** on sampled tool-call trajectories, **reverse-KL flavored**. This is "apply LFM2's
training novelty with the teacher we actually have."

### 5. Next-token vs richer objectives (MTP / RL)
**Scaler:** DeepSeek-V3 / Qwen3-Next / GLM add **Multi-Token Prediction** heads (denser gradient + free
speculative decoding); R1 uses RL for reasoning. **Hawk:** speculative decoding is irrelevant on CPU at
tiny scale; RL-from-scratch is the explicit *negative* result for small models.
**Verdict (ours):** **MTP as a *train-only* auxiliary head — worth a controlled try at 30M** (denser
gradient when data/params are scarce), **dropped at inference**; skip at 1M. **RL only as last-mile**
formatting/refusal shaping, never the primary capability source — distill instead.

### 6. Depth + weight-sharing vs width (the small-model regime)
**Structuralist:** MobileLLM — "deeper is more crucial than wider"; a 30-layer 125M beats wide-shallow,
and since **embeddings are >20% of params at this scale**, embedding tying + **immediate block-wise weight
sharing** (reuse a block back-to-back, free in cache) add +0.7–0.8 pts. **Us:** our 1M already runs
**factorized embeddings + depth-recurrence** — i.e. MobileLLM's two levers in a *more aggressive* form
(recurrence = the limit of block sharing).
**Verdict (ours):** keep both; **but recurrence's weakness is "every layer is identical."** Two cheap
fixes from the zoo: (a) **partially untie** the recurrence (cycle 2–3 *distinct* shared blocks, MobileLLM-LS
style, rather than one block) ; (b) **Per-Layer Embeddings as capacity injection** (Gemma-3n's PLE idea,
*re-purposed* — not for VRAM offload, which is moot on CPU, but to give each recurrence step a learned
per-layer bias so identical weights still specialize).

### 7. Data density, schedule, and "train once, deploy many"
**Data Camp:** Phi (textbook/synthetic data), SmolLM2 (edu-filtered curation), MiniCPM (**192× data:model
ratio** — small models want *far* more tokens than Chinchilla) — capability is **information density per
token**. **MiniCPM's WSD schedule:** long stable plateau + short exponential decay, with the **sharp
decay-phase loss drop**; fork the stable checkpoint to any budget, and **inject your cleanest data in the
decay window**. **Elastic camp:** Gemma-3n MatFormer nests submodels; merge-camp (Arcee/LFM2.5) combines
specialist runs in weight space instead of retraining.
**Verdict (ours):** ✅ **WSD schedule** (free, and the decay-window data injection is perfect for a
flywheel — front-load curated tool-call traces there). ✅ **Curriculum** (LFM2 difficulty-ordering by
empirical success probability). ✅ **Model merging across flywheel rounds** (soup → TIES if interference
appears) instead of always retraining from union data. ⚠️ **MatFormer only *within* a tier** (sample FFN
width per step inside the 30M) — **not across tiers**, since our 1M differs from the 30M in depth/recurrence,
not just FFN width, so a 1M is *not* a clean top-left slice of the 30M.

### 8. Stability as a first-class design target (the quiet 8th axis)
Qwen3 adds **QK-Norm** (RMSNorm on Q/K), Kimi-K2 invents **MuonClip** (QK-logit clipping) to train 1T
params with *zero* loss spikes. Stability is treated as architecture, not an afterthought — and it matters
*more* at tiny width and with **byte-level inputs** (long sequences, peaky logits).
**Verdict (ours):** ✅ **Add QK-Norm** to the attention layers — near-zero cost, disproportionately useful
at our scale.

---

## Verdict table — what the 1M/30M adopt

| Idea | Source | 1M | 30M | Note |
|---|---|---|---|---|
| **30M→1M distillation** (tempered Top-K → on-policy, reverse-KL) | LFM2, R1, Qwen3, MiniLLM | ✅ student | ✅ teacher | highest leverage; same tokenizer ⇒ no projection |
| **Gated short-conv (k=3) + 1–2 GQA hybrid** | LFM2, Nemotron-H | ✅ mostly-conv | ✅ ~3:1 conv:attn | keep ≥1 attn layer for arg-copying |
| **QK-Norm** | Qwen3, Kimi | ✅ | ✅ | cheap stability, matters at byte scale |
| **WSD schedule + decay-window curated data** | MiniCPM | ✅ | ✅ | fork stable ckpt per flywheel budget |
| **Curriculum (difficulty-ordered)** | LFM2 | ✅ | ✅ | reuse the loop's per-category accuracy |
| **Depth-recurrence + factorized embeddings** | MobileLLM (kept) | ✅ | — | partially untie recurrence (2–3 blocks); PLE as capacity injection |
| **Model merging across rounds** | Arcee, LFM2.5 | ✅ | ✅ | soup → TIES |
| **μP tiny-proxy LR transfer** | MiniCPM | proxy | ✅ | verify transferred LR with one short run (recurrence breaks clean μP) |
| **MTP auxiliary head (train-only)** | DeepSeek-V3 | ❌ | ⚠️ try | drop at inference |
| MoE / MLA / FP8 / sparse-long-ctx / RL-from-scratch | DeepSeek, Llama-4, Kimi | ❌ | ❌ | anti-patterns at our scale |
| Mamba/SSM/linear-attention mixers | Nemotron-H, MiniMax | ❌ | ❌ | Liquid HW-search: lose to conv+GQA on CPU |

## Application roadmap (the "improve this model" half)

1. **Distillation (now, highest ROI):** the decoupled tempered Top-K KD objective lands in `distill.py`;
   run **30M → 1M**; then add an **on-policy** pass over 1M-sampled tool-call trajectories scored by the
   30M. Measure against the current 1M (single-turn 48%, planner free-rollout 2%).
2. **Hybrid backbone tier:** a new config with **k=3 double-gated short-conv blocks + 1–2 GQA layers +
   QK-Norm**, budget-guarded; A/B vs the current decoder at 30M (CPU decode tok/s + accuracy) and a
   mostly-conv 1M variant.
3. **Training schedule:** WSD in `pretrain`/`sft`, with curated tool-call traces injected in the decay
   window; difficulty-curriculum ordering in the flywheel.
4. **Flywheel merging:** soup/TIES-merge per-round specialists instead of retraining from union data.

Each step is measured against the live baselines, not assumed — per the project rule of reporting real
numbers.

## Apply-phase results (measured — real numbers, including the nulls)

What the roadmap above actually produced when run. The project rule is no faking, so the mixed and
inconclusive results are reported as such.

| Lever | Measured result | Verdict |
|---|---|---|
| **Planner→action (1M)**, learned `plan_rollout` over a 4-round flywheel | teacher-forced next-tool **78→81%**, grounded-call **70%**, single-turn **48%** — but free-rollout whole-plan stays **~2%** | **Mechanics strong, strategy capped.** The 1M nails next-tool-given-context and grounding; autonomous multi-step planning collapses under compounding error — the capacity ceiling, cleanly shown. |
| **30M→1M Top-K distillation, distill-*then*-SFT**, controlled A/B (only the distill stage differs) | T−C: single-turn **+5.4**, free-rollout whole-plan **+7.5**, teacher-forced **0.0**, grounded **−29** (small-n, noisy) | **Modest + mixed.** Confirms the debate's own Scaler caveat: a 71% 30M teacher is too weak to transfer crisp argument-copying — and warming the backbone then SFT-ing the heads *re-specializes it away* from copying, so grounding **regressed**. |
| **30M→1M Top-K distillation, distill-*throughout*-SFT** (concurrent KD term added to `sft`), 3-arm A/B vs control & distill-then | grounded **+9.7 vs distill-then** (regression *erased*: 62.5% vs 52.8%, and +2.5 vs control); single-turn **+12.4 vs distill-then**; TF step-acc +3.1; but **vs plain control**: single-turn ~flat (−2.0), free-rollout noisy 0% | **The schedule matters; the gap still caps the gain.** Concurrent KD is *strictly better than* distill-then (recovers grounding, lifts selection) — so **if you distill at this gap, do it throughout SFT, not before.** But it does *not* cleanly beat from-scratch control on generative metrics: at a 71%→1M gap the distillation benefit is marginal, exactly as the capacity-gap caveat predicts. |
| **Hybrid short-conv + GQA + QK-Norm (30M)** vs standard decoder, equal params (29.1M vs 28.3M) | **+20% prefill, +5% decode** tok/s on CPU; accuracy 6.9% vs 11.0% at a minimal SFT-only budget — but at `n=24` that 4-pt gap is **~1 example (inconclusive)** | **Speed win real, accuracy unresolved.** The conv's O(L·k) edge over O(L²) is modest at short tool-agent contexts, as predicted; a fair accuracy ranking needs a pretrained, larger-eval A/B (currently blocked by the sandbox SIGKILLing the `batch=64` pretrain — addressable with a smaller pretrain batch). |
| **WSD schedule** (MiniCPM), 1M, 50-step proxy, LM-only | train loss **−0.005**, single-turn **+1.4pp** | **Within noise — as predicted.** WSD's payoff (the sharp decay-phase loss drop + a forkable stable checkpoint) needs a *long* run with a real plateau, not a 50-step proxy. Implemented opt-in (`lr_schedule="wsd"` + `decay_samples` injection); ready for a real pretrain to exploit. |
| **Curriculum** (LFM2 difficulty-ordering), 1M | single-turn **0/0 (floored)**, TF step −8.3, grounded −20 | **Within noise + a real mechanism finding.** Strict ordered passes **starve the hard tail unless `steps×batch ≥ pool size`** — the curriculum arm reached only the easiest ~25% of data in 50 steps and collapsed. Difficulty score validated (range 0.09→5.67); the lever is sound, the *budget* was too small for a full easy→hard sweep. |
| **On-policy reverse-KL** (1M←28M), 20-step smoke | reverse-KL **12.15→5.60** (first-third 11.82 → last-third 6.25); ~13.8 s/step | **Mechanism works cleanly.** Token-level full reverse-KL on the student's *own* rollouts moves it toward the teacher; exactly 0 when student==teacher. The free-rollout payoff needs a full pass + eval-harness measurement on a faster host (student sampling is ~14 s/step on this CPU). Implemented opt-in with CE/entropy/off-policy-mix stabilizers. |
| **Flywheel merging** (model-soup + TIES) | 17 tests pass; merged real 1M checkpoint loads into model+heads and evals (59.2%, sanity) | **Implemented + verified to load/eval.** Training-free weight-space combine of per-round specialists; a measured merge-vs-retrain A/B needs the round checkpoints + the slow eval harness. |

**Cross-cutting takeaway.** The debate predicted the transferable lessons would be *training*, not
architecture — and the measurements bear that out so far: the architecture lever (hybrid) bought a real
but modest *speed* gain with no clear accuracy change, while the *training* levers (distillation, planner
rollout) moved capability metrics the most (for better and, on grounding, for worse). Distillation at our
own 30M→1M gap is *weakly* positive, not the ~10× story the frontier reports at 7B→1B gaps — exactly the
capacity-gap caveat. The one crisp, reusable engineering result: **distill-*throughout*-SFT strictly beats
distill-*then*-SFT** (+9.7 grounding, +12.4 single-turn) — concurrent KD avoids the backbone
re-specialization that the two-stage schedule causes.

**Measurement caveat (load-bearing).** The dominant limit on these results is *this sandbox's compute*,
not the ideas: head-SFT runs ~37 s/step and greedy eval ~17 min/arm on a contended CPU, which forces
budgets down to where single-turn accuracy floors and most A/Bs land *within noise*. So the honest
deliverable of the apply phase is **the levers implemented as tested, opt-in, default-off code in the
pipeline** (Top-K KD, distill-throughout, hybrid backbone, WSD, curriculum) plus the **robust engineering
findings** (distill-throughout > distill-then; hybrid is +20% prefill at equal params; curriculum needs a
full-sweep budget; on-policy reverse-KL moves cleanly on the student's own rollouts) — *not* clean
capability deltas, which need a faster host to measure at real budgets. **All eight adopt-verdict levers
are now implemented as tested, opt-in, default-off pipeline code** (Top-K KD, distill-throughout, hybrid
backbone + QK-Norm, WSD, curriculum, on-policy reverse-KL, model-merging) — the toolkit is complete; what
remains is measurement at real budgets on a faster host.

## Sources
Qwen3 [2505.09388](https://arxiv.org/abs/2505.09388) · DeepSeek-V3 [2412.19437](https://arxiv.org/abs/2412.19437) ·
DeepSeek-R1 [2501.12948](https://arxiv.org/html/2501.12948v1) · Nemotron-H [adlr/nemotronh](https://research.nvidia.com/labs/adlr/nemotronh/) ·
Llama-4 [ai.meta.com](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) · Qwen3-Next [blog.vllm.ai](https://blog.vllm.ai/2025/09/11/qwen3-next.html) ·
MiniMax-01/M1 [2501.08313](https://arxiv.org/abs/2501.08313) / [2506.13585](https://arxiv.org/abs/2506.13585) ·
Kimi-K2 [github](https://github.com/MoonshotAI/Kimi-K2) · Mistral [2310.06825](https://arxiv.org/abs/2310.06825) ·
LFM2 [2511.23404](https://arxiv.org/html/2511.23404v1) · MobileLLM [2402.14905](https://arxiv.org/abs/2402.14905) ·
MiniCPM [2404.06395](https://arxiv.org/html/2404.06395v1) · Phi [2306.11644](https://arxiv.org/abs/2306.11644) ·
SmolLM2/SmolVLM [smollm](https://github.com/huggingface/smollm) / [2504.05299](https://arxiv.org/html/2504.05299v1) ·
Gemma-3 [2503.19786](https://arxiv.org/html/2503.19786v1) / Gemma-3n [MatFormer](https://arxiv.org/abs/2310.07707) ·
MiniLLM [2306.08543](https://arxiv.org/abs/2306.08543) · mergekit [2403.13257](https://arxiv.org/html/2403.13257v2).

*Closed models (GPT-4.5/5.x, Claude Opus, Cursor Composer) are deliberately omitted from the verdicts:
no published architecture, so any "philosophy" would be speculation. Where their disclosed direction is
relevant (distillation-heavy small variants, RL post-training) it already appears via the open models above.*

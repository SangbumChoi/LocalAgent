# Experiments — why we ran each one

Every plot in [`figures/`](../figures) with its **motivation/hypothesis**, what it tested, and what
we learned. Read this as the lab notebook.

---

### 01 · Free-gen vs grounded decoding
**Why:** a from-scratch byte model learns tool-call *structure* fast — but can it produce *correct*
calls by free generation? Hypothesis: no, because it can't copy unseen slot values.
**Finding:** raw byte-gen ≈ 1% on held-out; **grounded constrained decoding = 100%** on the same
model. → grounding, not raw generation, is how a tiny model becomes reliable.

### 02 · Flywheel accuracy per category
**Why:** does iterating (generate → train → enrich) actually improve a tiny agent, and which
categories lag? **Finding:** accuracy climbs per round; web/planner/text saturate early, tool args
lag — telling us where to spend data.

### 03 · Pretrain loss
**Why:** sanity check that a <1M byte model trains stably from scratch on CPU. **Finding:** clean
next-byte loss decrease — the from-scratch premise holds.

### 04 · Single-turn vs multi-turn
**Why:** real agents are multi-turn (tool → response → follow-up). Can the model ground a follow-up
arg from a tool *response*? **Finding:** multi-turn reaches ~73% step accuracy once the heads are
trained on episode contexts — but only then (a frozen single-turn model transfers at 0%).

### 05 · 21-tool per category
**Why:** does adding the productivity/computer-use surface (21 tools) hold up? **Finding:** grounding
exact (0 misses) but per-tool *selection* spreads thin — first sign that selection is the limit.

### 06 · Parallel two-call turns
**Why:** people say "do X **and** Y" — one turn, two calls. Can a 1M model do it? **Hypothesis:** it
needs the head trained on the split *conjuncts*. **Finding:** 0→38% on 1M once we train on
conjuncts; the compounding (both calls must be exact) makes it intrinsically hard.

### 07 · Failure-driven flywheel
**Why:** generic enrichment is wasteful — does *mining the model's failures* and oversampling the
weak tools improve faster? **Finding:** 45→62% over 5 rounds; `run_command` 0→100%, but `git_commit`
stuck at 0% — data alone can't fix tools the 1M selector can't separate.

### 08 · Dataset size (62→71%)
**Why:** was the 1M *data-starved* or *capacity-limited*? Hold the model fixed, 4× the data.
**Finding:** 62→71% — partly data-starved. Data fixes the data-starved tools.

### 09 · Model size (1M vs 28M)
**Why:** the complement to 08 — hold data fixed, grow the model. **Hypothesis:** capacity fixes the
*confusable* tools that data couldn't. **Finding:** 28M beats 1M overall (+4) and transforms two-call
(+62). → **data and size buy different things.**

### 10–11 · Throughput & memory
**Why:** "runs on CPU/edge" is a core claim — measure it, and prove the KV cache matters.
**Finding:** KV-cache decode is ~3–4× faster than recompute; memory is dominated by params, KV cache
is tiny at these sizes.

### 12 · Tool retrieval at scale
**Why:** a fixed N-way classifier head can't scale to 100s–1000s of tools — does retrieval?
**Hypothesis:** yes, and indexing by *example usages* beats indexing by description on paraphrased
queries. **Finding:** retrieval indexes 1,350 tools in 0.27s; example-augmented recall@10 ≈ 80% vs
34% description-only. → retrieval is the selection architecture at scale.

### 13 · Distillation
**Why:** can a 28M teacher distill into the 1M student (cheaper than RL)? **Hypothesis:** logit-KD
helps. **Finding (honest negative):** on *deterministic* templated targets KD gives ~0 — hard-label
SFT is already optimal and softening hurts. → distillation suits ambiguous targets, not these.

### 14 · Coding tools by argument count
**Why:** intuitively, multi-argument tools (e.g. 3-arg `edit_file`) should be the *hardest* to
ground. **Finding (surprise):** the opposite — 1-arg 68%, 2-arg 88%, 3-arg 100%; args-exact is 94.6%.
More args = more distinctive query = easier. Grounding isn't the bottleneck; **selection among
confusable 1-arg tools is.**

### 15 · How much per-tool data does retrieval need?
**Why:** if you onboard a new tool, how many example invocations must you write? **Finding:**
description-only = 44% tool@1; **one example = 60% (+16 pts)**, saturating ~16. → a handful of
examples per tool is plenty.

### 16 · Tool-calling scenarios: MCP vs REST vs CLI vs SDK
**Why:** real tools come in different surface forms — MCP server schemas, REST endpoints, CLI
commands with flags, SDK method calls. **Hypothesis:** named-JSON args (MCP/REST) ground more easily
than CLI flags / SDK positional args where the value's role is implicit. **Finding (hypothesis
refined):** SDK **100%**, REST 92%, MCP 83%, **CLI 58%** (worst). The driver turned out to be
**argument *typing*, not the modality**: CLI's two-arg commands with bare *plain-string positionals*
(`docker_run(image, port)`, `kubectl_get(resource, namespace)`) are hard because an untyped string
grabs the wrong span; SDK was easy only because its args are typed (paths, quoted, enums).
→ **Lesson for tool authors: give each argument a `format`/`enum`/type (or quote its value) and
grounding is reliable on any surface form.** See `scripts/scenarios_eval.py` / `figures/16_*`.

---

## The throughline
Grounding is the easy part; **selection is the bottleneck**, and it's fixed by *capacity* and a
*re-ranker*, not by more data (which saturates fast). Data fixes data-starved tools; size fixes
confusable ones. A tiny model is made *reliable* by constrained decoding + retrieval, not by scale.

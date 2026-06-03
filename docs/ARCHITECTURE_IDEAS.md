# Architecture ideas — going past the open-source playbook

> We don't have to copy what the small-LLM repos do. This doc collects original structural ideas
> for LocalAgent, says which we **ship now** vs **propose**, and gives a recommendation. The
> central bet: at the sizes we care about, the *agent-as-controller* framing changes the
> architecture, and the *vocabulary* is the dominant design lever.

## 0. The reframe: at ~1M params, build a controller, not a chatbot

A 1M-param model cannot store the world. So stop trying. An agent's job is **routing** — given a
user turn + tool specs + memory, decide: *answer directly, call tool X with args, or abstain*.
That decision is mostly **pattern + control flow**, not knowledge. Knowledge is offloaded to
**tools** and the **memory/retrieval store** (which the bigger pipeline already has).

This gives each tier a distinct job rather than "the same model, smaller":

| Tier | ~params | Job | Vocab | Realistic ask |
|---|---|---|---|---|
| **ultra-tiny** | ~1M | **tool router / planner** | byte (256) | pick the right tool + args, abstain when no tool fits; little free-form text |
| **tiny** | ~30M | **agent** | BPE 32k | tool calls + short grounded answers + memory ops |
| **small** | ~90M | **capable local agent** | BPE 32k | multi-turn tool use + coherent text |

The eval that matters for ultra-tiny is **BFCL AST accuracy + irrelevance**, *not* perplexity.

## 1. The vocabulary tax (why ultra-tiny needs a different structure)

At 1M params a 32k×d embedding table is impossible — with d=128 it's already 4M params, 4× the
whole budget. So the embedding *is* the architecture problem at this scale. Three levers, all now
wired into `ModelConfig`:

- **Byte-level vocab (256).** Embedding table becomes negligible (256×embed_dim). Sequences get
  longer, but tool-routing turns are short. Tokenizer-free, trivially robust to any input.
- **Factorized embeddings (ALBERT-style).** Learn `vocab×embed_dim` then up-project
  `embed_dim→d_model`. Decouples vocab cost from model width. (`embed_dim` knob.)
- **Depth-recurrence / weight sharing (Universal Transformer + ALBERT).** Run `n_layers` blocks
  `n_loops` times with shared weights → *effective depth = n_layers × n_loops* at the param cost
  of `n_layers`. A 1M model gets depth-12 reasoning from 2 blocks. (`n_loops` knob, with a small
  per-loop embedding so a block knows its iteration.)

`ultra-tiny-1m` = byte(256) + embed_dim 64 + 2 blocks × 6 loops ≈ **0.98M params, effective depth 12.**
These three are **implemented and tested today.**

## 2. Proposed structural ideas (designed, not yet built)

### 2a. Dual output head: a **tool head** + a **text head**
Instead of forcing the tiny model to emit valid JSON token-by-token, give it two heads off the
final hidden state:
- a **tool head** = a classifier over the (small) set of available tools + an "no-tool/answer"
  class;
- **argument slots** filled by a **pointer/copy** mechanism over the prompt (copy `Paris` from the
  user turn) plus typed value heads, validated against the tool's JSON schema.

This turns "generate syntactically-valid function calls" — hard for a 1M model — into
classification + copying, which tiny models do well. The text head handles free-form. Pairs
naturally with the byte backbone. *This is the highest-leverage original idea here.*

### 2b. Grammar-constrained decoding, structurally enforced
Tool-call spans are decoded through the tool's **JSON-schema grammar** (a finite-state mask over
next-token logits). The model *cannot* emit an invalid call — correctness is a property of the
decoder, not something the tiny model must learn. Combined with 2a, the model picks the tool and
the decoder guarantees the shape.

### 2c. KV-free backbone option (SSM / linear attention)
A `backbone: attn | ssm` switch. For multi-turn agents on edge/NPU, an SSM (Mamba/RWKV-style)
gives **constant memory per step** (no growing KV cache) — attractive for long tool traces and
the memory module. Attention stays the default; SSM is the edge/long-context variant. Reserved as
a config knob.

### 2d. Parameters = skills, external store = knowledge
Lean fully into retrieval: the weights encode *control/skills*, a kNN/RETRO-lite store holds
*facts*. This is already half-present via `agent/memory.py` + the data flywheel; the structural
add is a retrieval *head* that conditions generation on fetched snippets, so the tiny model never
needs to memorize.

### 2e. Micro-MoE for the `small` tier (optional)
Many tiny FFN experts, top-1 routed. Raises capacity at ~constant inference cost. Likely overkill
/ unstable below ~30M; only considered for `small` and clearly marked optional.

## 3. Recommendation

1. **Ship now (done):** the three vocab/structure levers + the three-tier scheme. They are what
   make a sub-1M agent even plausible.
2. **Build next (Phase 4 alongside SFT):** **2a + 2b** — the dual tool-head + grammar-constrained
   decoding. This is where a small, original design beats "shrink a Llama," because it converts the
   hardest tiny-model task (valid tool calls) into easy ones (classify + copy + masked decode).
3. **Keep as switches:** **2c** (SSM backbone) for the edge/NPU export story, **2d** retrieval head
   for knowledge offload. **2e** stays optional.

Everything here is additive to the existing pipeline — same `Conversation` schema, same
train/eval/export stages, same data flywheel. See `docs/ROADMAP.md` for where each lands.

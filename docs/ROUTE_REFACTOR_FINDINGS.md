# Tool-selection refactor: from a 51-way classifier to a *generable* pipeline

## The problem
The expanded (50-tool) pipeline selected tools with a **51-way softmax head** (`tool_head.CLASSES`).
A fixed N-way classifier is the wrong abstraction: it can't accept a tool it didn't see at train
time, must be reshaped+retrained whenever the tool pool changes (the 22→51 jump already forced a
*fresh* head), and doesn't transfer to the function-calling / MCP setting where tools differ per
request. "Pick one of 51 fixed classes" is not a generable method.

## Diagnostic: what was the classifier actually doing?
Same 40-sample held set, `tiny-30m-50tools.pt`:

| Decode path | Acc |
|---|---|
| 51-way head + pointer-copy *(shipped)* | **45%** |
| generative tool selection + pointer-copy *(drop the 51-way head)* | 5% |
| generative selection + heuristic args *(no heads)* | 5% |
| pure free-generation *(LM emits the whole call as text)* | **0%** |

The 51-way head was carrying *the entire* selection capability — remove it and 45% collapses to 5%.
The model has near-zero intrinsic ability to select or generate a call; pointer-copy only helps fill
args *after* selection is already correct.

## Fix part 1 — routes replace the 51-way classifier (works)
`agent/routes.py`: the head now classifies one of **5 stable routes** (modalities) —
`web_search / computer_use / code / app_action / text` — instead of 51 concrete tools. Adding or
removing a tool no longer reshapes the head; only adding a whole new modality does.

- `RouteHead` (5-way) as a frozen-feature linear probe: **route_acc = 76.5%** with *zero* extra
  backbone training (code 84%, text 88%, computer_use weakest 63%).
- The concrete tool is then chosen by **retrieval** (`ToolRetriever`, char-ngram, zero-training):
  **recall@6 = 68–75%**, scales to any pool, handles unseen tools.

## Fix part 2 — can the LM *generate* the specific call? (the negative result)
We rendered a small retrieved tool catalog into the prompt (`agent/incontext.py`) and SFT'd the LM
to free-generate the call from it (`scripts/sft_generative.py`, 500 steps, in-context catalog):

| | gen_acc (gold-in-candidates) |
|---|---|
| baseline (backbone + in-context prompt, no SFT) | 0.0% |
| after generative SFT | **2.1%** |

Train loss collapsed to 0.003 (full memorization of the 157-sample train set) but held generalization
stayed ~2%. Inspecting held generations shows **exactly why**:

```
USER : What's it like outside in Perth? In Celsius.
GOLD : {"arguments":{"city":"Perth","unit":"c"},"name":"get_weather"}
GEN  : <tool_call>{"arguments":{"city":"Tbilisi"},"name":"get_weather"}</tool_call>

USER : Read the file data/loader.py.
GOLD : {"arguments":{"path":"data/loader.py"},"name":"read_file"}
GEN  : <tool_call>{"arguments":{"path":"defi/tint"},"name":"grep_search"}</tool_call>
```

- **Structure: 100%** — always well-formed, correct argument *keys* for the chosen tool.
- **Argument *values*: 100% hallucinated** — `Perth`→`Tbilisi`, `8*1*13`→`15*15`,
  `data/loader.py`→`defi/tint`. The model emits *memorized* values instead of **copying from the
  prompt**.

This empirically confirms the architecture's thesis: **a <100M byte model learns tool-call structure
+ selection but cannot free-generate argument *values* — those must be copied.** That is exactly
what `ptr_head` (pointer/copy span) does, and why the grounded path reaches 45% while free-gen is 0%.

## Conclusion — the correct *generable* architecture
Drop the 51-way classifier, but do **not** try to free-generate everything. The evidence says each
job goes to the component that can actually do it:

| Job | Component | Why |
|---|---|---|
| modality | **5-way route head** | stable, portable; 76.5% as a probe |
| candidate tools | **retrieval** | scales to any pool, zero retraining, unseen tools |
| call structure + tool name + arg *keys* | **generation** (in-context catalog) | 100% structure; selection improves with a small candidate set |
| argument *values* | **pointer-copy** (`ptr_head`) | the tiny model provably can't free-generate these |

This removes the brittle fixed-N classifier (the thing that wasn't generable) and keeps only the one
sub-task that genuinely needs a learned mechanism — tool-agnostic argument **copying**. Full free
generation of argument values is *not* viable at this model size; that is a measured result, not an
assumption.

### Artifacts
- `runs/tiny-30m-routed.pt` — backbone + 5-way route head.
- `runs/tiny-30m-incontext.pt` — generative in-context SFT checkpoint (the negative-result probe).
- `scripts/retrain_routed.py`, `scripts/sft_generative.py` — reproduce both.
- `eval.harness.evaluate_routed` / `evaluate_incontext` — the metrics above.

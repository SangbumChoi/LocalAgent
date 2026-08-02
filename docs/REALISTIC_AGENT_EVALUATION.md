# Realistic agent evaluation and data policy

This project targets a text-first WebGPU agent with a small (<100M) decoder.  Real deployments
still need to be tested against the environments people use: mobile UI control, browser actions,
desktop workflows, and stateful API/MCP tools.  The source-linked inventory is
[`configs/data/realistic-agent-eval.catalog.yaml`](../configs/data/realistic-agent-eval.catalog.yaml).
It is deliberately a catalog, not a downloader: every acquired byte must be recorded in a local
provenance manifest with an upstream revision, byte count, and SHA-256.

The current catalog contains 40 source-linked rows (four train-eligible and 36 evaluation or
restricted) and has a canonical SHA-256 fingerprint recorded by the preflight command below.
The fingerprint is generated from the canonical catalog by the preflight command below.
The post-freeze public-source audit is kept separately in
[`configs/data/realistic-agent-eval.supplemental.yaml`](../configs/data/realistic-agent-eval.supplemental.yaml).
It adds eight high-value sources—Computer Agent Arena, CUA-Lite AgentNet, OSWorld 2.0
trajectories, EnterpriseOps-Gym, MCPMark, ToolSandbox, AndroidWorld, and BrowserGym—with explicit
license, split, runtime, and WebGPU-projection policies. These entries are catalog-only until an
exact revision and acquisition receipt are frozen; they do not silently become training data.

Run the read-only readiness report before acquiring or evaluating anything:

```bash
PYTHONPATH=src python scripts/realistic_agent_preflight.py
```

The report currently identifies the four local text-first adapters as runnable and all 36
environment/evaluation rows as blocked by their pending integration status (for example, no
`adb`, Docker, VM, or upstream BrowserGym checkout).  `--strict` intentionally exits non-zero
until those external runners are installed and pinned; this is a readiness gate, not a benchmark
score.

The current inventory also tracks recent realism upgrades: AndroidDaily (daily-use closed-source
mobile tasks), Windows Agent Arena (Windows desktop workflows), WebChoreArena (memory-heavy
browser chores), and TimeWarp (cross-version web UI drift).  They are evaluation-only until their
runtime dependencies, task revisions, and licensing terms are pinned.

### Workshop/publication gate

The catalog preflight is necessary but not sufficient for a workshop claim.  The stricter
[`scripts/workshop_gate.py`](../scripts/workshop_gate.py) requires explicit native receipts for
AndroidWorld, BrowserGym/MiniWoB, OSWorld, AgentNet, ToolSandbox, MCPMark, and EnterpriseOps-Gym;
it also requires a hardware-WebGPU capability/latency receipt, both transfer and no-transfer
weight reports, and a public model/demo manifest.  It never treats a protocol bridge, a synthetic
state loop, SwiftShader, or a local checkpoint path as a pass:

```bash
PYTHONPATH=src python scripts/workshop_gate.py --strict
```

The command without supplied receipts exits non-zero with nine blocking requirements.  Supplying
the verified native WebGPU, full BrowserGym, and public-artifact receipts reduces this to six:

```bash
PYTHONPATH=src python scripts/workshop_gate.py --strict \
  --webgpu-receipt docs/paper/results/raw/m40-webgpu-native-capability-notion-v1.json \
  --native-receipt browsergym_miniwob=docs/paper/results/raw/m43-browsergym-native-adapter-full-model-eval-v1.json \
  --weight-report docs/paper/results/raw/m25-weight-transfer-ablation-v1.json \
  --public-artifact-manifest docs/paper/results/raw/public-model-demo-manifest-v1.json
```

The six remaining native benchmark receipts are still absent.  The public artifact manifest now
verifies the already-live 28.32M-parameter byte model and static WebGPU Space, but it does not
prove native OS/emulator/MCP control or task success.  That is intentional:
four train adapters and the tracked offline receipts are useful progress, but they do not prove
those native task outcomes.  Once a native runner produces a receipt, it
can be supplied as `--native-receipt BENCHMARK_ID=PATH`; the receipt contract requires explicit
environment execution, official split verification, task count, and success rate.  The gate is
therefore an auditable workshop decision, not a claim that the current model has passed.

The actual SFT checkpoint has also been exercised through the local deployment path with
[`scripts/deploy_smoke.py`](../scripts/deploy_smoke.py).  The pinned
[`m24 receipt`](paper/results/raw/m24-local-deployment-smoke-v1.json) records 10 realistic prompts
covering browser, computer, calculator, text, and email intent.  It selected the expected tool on
4/10 prompts (browser 1/3, computer 1/4, productivity 1/1); the text prompt incorrectly emitted a
tool.  The handlers only echo calls, so this is a useful regression smoke and a concrete failure
sample, not browser/OS/email capability evidence.

The HF-format exporter was corrected before any publication attempt.  A local export of the same
checkpoint now includes the recorded 16K BPE tokenizer, checkpoint/tokenizer hashes, safetensors,
and `tool_head`, `ptr_head`, `route_head`, `dense_selector`, and `selector_proj`.  The
[`m26 receipt`](paper/results/raw/m26-hf-local-export-v2.json) binds all five file hashes and marks
`published: false`: this environment is not authenticated to Hugging Face, so no public model URL
is claimed.  That receipt describes the newer local 10.52M BPE export.  Separately, the older
public `danelcsb/localagent-tiny-30m-byte` model and `danelcsb/localagent-webgpu` Space are live
and hash-bound by [`public-model-demo-manifest-v1`](paper/results/raw/public-model-demo-manifest-v1.json).
The public deployment is a 28.32M-parameter byte model; it must not be described as the m46 BPE
continuation or as evidence of native Android, desktop-VM, MCP-server, email, or Notion task
success.

The first native checkpoint-in-loop browser probe is now recorded in the [`m29 receipt`](paper/results/raw/m29-browsergym-native-model-eval-v1.json).
It executed the pinned BrowserGym/MiniWoB environment for all 240 episodes (60 task variants,
four fixed seeds) with live accessibility-tree observations.  This is a real environment result,
but only a one-step diagnostic: the current SFT checkpoint abstained on all 240 episodes, produced
zero grounded actions, and achieved 0/240 success.  It is explicitly not an official multi-step
BrowserGym score and does not support visual, WebArena, mobile-emulator, desktop-VM, or real-account
email/Notion claims.

### Browser-context contract adaptation (m30)

The native probe exposed a concrete distribution mismatch before any browser reward could be
learned: the SFT checkpoint was trained mostly on short action prompts, while the runner supplied a
goal followed by a live accessibility tree.  The tracked
[`train_browser_context_adapter.py`](../scripts/train_browser_context_adapter.py) addresses only
that interface contract.  It filters single-turn synthetic computer-use rows, adds deterministic
quoted element names and an abstention instruction, then performs 300 low-rate backbone updates and
1,000 route/selector probe updates.  It never reads BrowserGym goals, screenshots, task plans, or
labels.

On ten held-out projected rows (one each for click, double-click, drag, key press, move, open-app,
screenshot, scroll, type, and wait), route accuracy moved from `40%` to `100%`, tool accuracy from
`30%` to `100%`, and exact argument accuracy from `20%` to `60%`.  The complete hash-bound receipt is
[`m30`](paper/results/raw/m30-browser-context-adapter-v1.json).  This is evidence that the
observation-contract adapter is learnable, not a native BrowserGym result; the remaining argument
errors and the 240-episode m29 abstention result still block any browser-agent claim.  The native
runner now emits quoted accessibility names so a future run evaluates the same contract.

The replayed train-only adapter is recorded in [`m42`](paper/results/raw/m42-browser-context-adapter-replay-v1.json).
It reused 589 synthetic computer-use rows for 300 low-rate backbone updates plus 1,000 route and
selector-head updates.  On ten disjoint projected rows, route accuracy improved `40% → 100%`,
tool accuracy `30% → 100%`, and exact arguments `20% → 60%`.  This is a contract-transfer result,
not BrowserGym training; no BrowserGym goals, screenshots, task plans, or labels were read.

The complete pinned native split is now recorded in the [`m43 receipt`](paper/results/raw/m43-browsergym-native-adapter-full-model-eval-v1.json).
It ran the exact 240-episode plan (60 task variants, four fixed seeds, ten model steps per
episode) with BrowserGym `0.14.3`, MiniWoB++ at the pinned revision, Playwright `1.44.0`, and
Chromium `125.0.6422.26`; `official_split_verified` is true.  The train-only accessibility
contract adapter produced `5/240` success (`2.08%`): two `click-button` and three
`stock-market` episodes.  This is a reproducible native text/accessibility-tree ablation, not a
visual-agent, WebArena, real-account, or leaderboard score; the compact tracked receipt retains
per-task and per-seed aggregates and hashes the full local case log.  The unchanged base
checkpoint remains the m41 `0/240` baseline.

### Public Mind2Web trajectory continuation (m44)

To validate the public-data path independently of the older bounded `train_10` shard, the Hugging
Face dataset-server `rows` endpoint was queried at the pinned Mind2Web revision with a bounded
request for eight `train` records. Five records passed the positive-grounding and operation
allowlist, yielding 20 normalized Conversation variants. Four parent records (16 rows) were used
for continuation and one unseen parent (four rows) was held out; parent IDs and typed slot values
were disjoint. The acquisition, filtered-source, and normalized-output hashes are in the
[`m44 acquisition receipt`](paper/results/raw/m44-mind2web-acquisition-v1.json), while the
training metrics and child checkpoint hash are in the [`m44 continuation report`](paper/results/raw/m44-mind2web-trajectory-continuation-v1.json).

Starting from the 10.52M BPE WebGPU parent, 32 low-rate SFT updates plus 128 route/selector probe
updates reduced held-out mean loss `2.2349 → 1.6979` and raised assistant-token accuracy
`66.14% → 74.21%`. Sequence exactness remained `0/4`; the selector reached `7/11` tool
decisions (`63.64%`) and route accuracy reached `34/34`. These are text-first held-out
train-record metrics, not an official Mind2Web test score.

The paired [`m46 weight-transfer analysis`](paper/results/raw/m46-weight-transfer-analysis-v1.json)
shows why the pretrained backbone remains deployable: configuration, tokenizer, and all 51 shared
tensors are compatible; the attention/mixer, FFN, and embedding groups moved only `0.37%`, `0.45%`,
and `0.93%` in relative L2, respectively, while action heads moved `69.19%`. This supports a
small-learning-rate backbone plus larger head-rate recipe, but does not prove transfer is optimal;
the existing matched no-transfer ablation remains the required quality control.

The child was also exercised in a ten-episode pinned BrowserGym/MiniWoB canary using the explicit
`realistic_browser` pool (`web_click`, `web_type`, `web_select`). The [`m45 receipt`](paper/results/raw/m45-browsergym-native-realistic-toolpool-canary-v1.json)
records `0/10` success. This is a diagnostic only: Mind2Web backend-node IDs are not MiniWoB's
live DOM IDs, so the canary does not measure cross-site grounding or justify an official BrowserGym
score. The native runner defaults to the legacy `standard` pool and now exposes the realistic pool
explicitly to prevent silent vocabulary mismatch.

### Larger bounded public Mind2Web continuation (m46)

The next dataset-server request covered 16 additional `train` records. Twelve passed the same
fail-closed grounding rule after truncating each sequence to 12 actions, producing 48 normalized
rows. Nine parent records (36 rows) were used for training and three unseen parents (12 rows) for
evaluation; the acquisition receipt records zero parent and typed-slot overlap. Starting again from
the same 10.52M parent, 64 backbone updates and 256 frozen-feature route/selector updates raised
held-out assistant-token accuracy `65.75% → 78.73%` and reduced mean loss `2.2328 → 1.4992`.
Route accuracy reached `66/66`, selector top-1 reached `55/63` (`87.30%`), and sequence exactness
remained `0/12`. This is stronger public-train evidence than m44, but it is still not an official
Mind2Web test score, native browser success, or screenshot-grounding result.

### Bounded public AgentNet computer-use continuation (m47)

The public [OpenCUA AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) Ubuntu trajectory
file was acquired at revision `d76ee50a63fad81cfdbe576416757d7c2091ed50` using an explicit
`0..8,388,607` byte range. The bounded prefix contains 40 complete parent tasks; a deterministic
parent-disjoint split uses 32 tasks for training and eight for evaluation. The new
[`ingest_agentnet_text.py`](../scripts/ingest_agentnet_text.py) adapter keeps task text and a
bounded textual screen observation, maps click/double-click/drag/keyboard/type/scroll/cursor/wait
actions into existing computer-use tools, and drops screenshots plus termination/triple-click
markers. The acquisition and projection identities are in the
[`m47 acquisition receipt`](paper/results/raw/m47-agentnet-acquisition-v1.json) and
[`projection metadata`](paper/results/raw/m47-agentnet-projection-metadata-v1.json).

Starting from the 10.52M BPE mobile-dispatch parent, 32 SFT updates on 513 projected train rows
reduced the source-disjoint 133-row teacher-forced mean loss `3.7804 → 2.6121` and raised token
accuracy `47.09% → 58.89%`. Sequence exactness remained `0/133`; route accuracy was `99.25% →
100%`, and the frozen-feature selector top-1 diagnostic moved `5.26% → 12.03%`. This is a
text-only desktop-action continuation, not an official AgentNet validation score, AgentNetBench,
OSWorld, screenshot-grounding, or native desktop result. The paired
[`m47 weight-transfer report`](paper/results/raw/m47-agentnet-weight-transfer-v1.json) found equal
configuration/tokenizer and 51 compatible tensors: attention/mixer, FFN, and embedding movement
was `0.29%`, `0.35%`, and `0.49%`, with action heads unchanged because this run intentionally
isolated backbone SFT. The result supports warm-start compatibility, not a claim that transfer
is optimal.

The matched [`m48 transfer ablation`](paper/results/raw/m48-agentnet-weight-transfer-ablation-v1.json)
uses the same rows, tokenizer, configuration, seed, 32 updates, and learning rate with a fresh
backbone control. After continuation, the warm-start arm reached `58.89%` held-out token
accuracy and mean loss `2.6121`, versus `21.19%` and `8.5930` for the fresh-backbone arm; both
arms remained at `0/133` sequence exactness. This isolates a strong initialization effect in this
bounded pilot, while leaving the no-transfer and native-runtime limitations explicit.

### AgentNet text-observation/action projection evaluation (m62)

The retained eight-parent evaluation projection was then run through the actual LocalAgent
constrained decoder and the repository's AgentNetBench-compatible scorer.  This is a stronger
checkpoint-in-the-loop test than teacher forcing: each of the 133 retained action rows is decoded
from its task plus bounded textual observation, grouped back into its parent trajectory, and
scored for action type, coordinates, text, keyboard, scroll, and sequence alignment.  The
[`m62 receipt`](paper/results/raw/m62-agentnet-text-projection-eval-v1.json) verifies all eight
parent IDs and records the exact source/model/projection hashes.

The warm-start SFT child gets `75%` first-action-type agreement, but mean trajectory score is
approximately zero, exact trajectory success is `0/8`, and coordinate/text/keyboard/scroll
families receive no useful credit.  The matched random-backbone child gets `0%` first-action-type
agreement and zero score.  This separates two facts: pretrained initialization transfers a
coarse action prior, while screenshot-dependent grounding and long-horizon state alignment do
not transfer to text-only WebGPU inference.  The run is explicitly not official AgentNetBench,
native desktop, OSWorld, or screenshot-grounding evidence; termination and triple-click source
actions dropped by the text projection are outside the score.

### Computer Agent Arena metadata audit (m49)

The public [Computer Agent Arena](https://huggingface.co/datasets/xlangai/computer-agent-arena)
metadata file was downloaded at revision
`897b9f45287c516a44f9e79879b14bc3c1bc5b0a` and hash-bound in the
[`m49 receipt`](paper/results/raw/m49-computer-agent-arena-metadata-v1.json).  The snapshot has
4,641 unique trajectories and 99,765 steps; 76,942 steps reference screenshots, 83,007 contain
textual thought, and 89,144 contain a conservatively recognized computer-use primitive.  The
metadata records a 57.90% human-evaluation correctness rate across the published trajectories.
The 50,609,777-byte JSONL was read, but no screenshot archive was downloaded or opened.

This is a source and modality-coverage audit, not a Computer Agent Arena score: the benchmark is
evaluation-only, screenshots are omitted from the text-first WebGPU path, and the rows were not
used for SFT.  The result makes the visual-grounding gap measurable rather than silently treating
thought text as an observation.  A future visual model must use the pinned image archive and a
task-disjoint protocol before any image-grounded or native desktop claim is made.

### Computer Agent Arena instruction-only checkpoint probe (m50)

The current 10.52M-parameter BPE WebGPU checkpoint was then evaluated on 128 deterministic,
source-disjoint first-action cases using only each public task instruction.  The probe is recorded
in the [`m50 receipt`](paper/results/raw/m50-computer-agent-arena-instruction-probe-v1.json) and
implemented by [`evaluate_computer_agent_arena_text.py`](../scripts/evaluate_computer_agent_arena_text.py).
The route head correctly identified `computer_use` on `128/128` cases, but exact local-tool
selection was only `5.47%` and action-family accuracy was `7.03%`; the model abstained on `9.38%`
of cases.  It over-produced `screenshot` for pointer tasks, which is the expected failure mode when
the model is asked to ground a coordinate action without an image or accessibility tree.

This is a deliberately leakage-safe action-prior diagnostic.  It does not use the published
thought traces, screenshots, later trajectory state, or action arguments, and it is not an official
Computer Agent Arena, AgentNetBench, visual-grounding, or native desktop result.  The result is
useful for model design: the route gate transfers, while grounding requires a visual/accessibility
encoder and a stateful action loop before further SFT or RL can be expected to improve real tasks.

## Bounded public Mind2Web training continuation

The public [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) training split was streamed
at revision `17ece8eb89862368edc0cc806acee6fca5163474` without placing raw data in Git.  Of the
first eight streamed records, five passed the adapter's fail-closed positive-grounding rule; they
normalized to 10 deterministic Conversation rows, 116 grounded tool calls, and the
`web_click`/`web_type`/`web_select` capability set.  The complete byte/hash/training receipt is
[`mind2web-public-train-sample-v1.json`](paper/results/raw/mind2web-public-train-sample-v1.json).

Sixteen BPE-tokenizer SFT updates from the corrected 10.5M mobile parent reduced mean loss from
`2.2790` to `1.7595` and raised assistant-token accuracy from `63.97%` to `74.29%`; exact
assistant trajectories remained `0/10`.  The transfer audit found 51 shared tensors, equal model
configuration and tokenizer identity, no shape additions/removals, and frozen action heads.  This
is legitimate public-data adaptation evidence, not a native Mind2Web score: the protected test
archive and browser environment were not executed.

### Held-out public Mind2Web continuation (m31)

The next public-data run downloaded only the pinned `data/train/train_10.json` shard, retained six
records whose complete action sequences had positive grounded backend-node IDs, and normalized 24
Conversation rows. Five source records (20 enriched rows) were used for training; the sixth source
record (four rows) was held out, with both parent-record IDs and typed slot values disjoint. Raw
Mind2Web bytes remain outside Git; the exact shard, filtered snapshot, normalized output, split
hashes, and manifest are bound in the [`m31 receipt`](paper/results/raw/m31-mind2web-public-train10-continuation-v1.json).

This run also closes a tool-catalog mismatch without changing the legacy 50-tool checkpoint shape:
[`REALISTIC_BROWSER_TOOLS`](../src/localagent/agent/toolset.py) extends the dense two-tower pool with
`web_click`, `web_type`, and `web_select`, and all three map to the stable `computer_use` route.
After 32 low-rate backbone updates and 256 frozen-feature head updates, held-out teacher-forced
token accuracy rose from `69.1%` to `80.0%`; the browser selector rose from `0/6` to `6/6` top-1
tool decisions. A generative replay was weaker—parseable calls `2/6` to `3/6`, exact tool names
`0/6` to `3/6`—because multi-turn context and argument copying still fail. This is public-train
generalization evidence, not an official Mind2Web test score or native browser execution result. The
extended 53-tool dispatch export also reproduced PyTorch route argmax and selector top-1 exactly
(`1.0` agreement; max score drift `1.26e-6`), proving that the new browser columns can be shipped
to a device without changing the backbone graph.

The adapted child was then exported through the browser bundle path.  The fp32/fp16 logits and
hidden graphs plus the distinct hidden-only action graph passed the hard parity gate; fp32 logits
drift was `6.91e-06`, fp32 hidden drift `5.66e-06`, and fp16 logits argmax agreement was `1.0`.
In the in-app browser, an explicit WebGPU session loaded the verified 10.5M bundle and the local
9-step mobile/email/Notion state suite produced schema validity `6/9` and closed-loop success
`4/9` (mobile `2/7`, productivity `2/2`).  The WASM control had the same quality result.  The
versioned receipt is [`m11-webgpu-mind2web-bpe-child.json`](paper/results/raw/m11-webgpu-mind2web-bpe-child.json);
no hardware throughput claim is made because timing was not collected in this run.

### Realistic browser-tool WebGPU bundle (m32)

The same held-out Mind2Web child was exported with the 53-tool
[`REALISTIC_BROWSER_TOOLS`](../src/localagent/agent/toolset.py) pool, including `web_click`,
`web_type`, and `web_select` while preserving the 10.5M backbone and the existing pointer-head
shapes.  The model graph, hidden-only action graph, serialized heads, tokenizer, and tool metadata
are hash-bound in the [`m32 receipt`](paper/results/raw/m32-webgpu-realistic-browser-tool-pool-v1.json).
The fp32 and fp16 export gates passed; the JavaScript grounding bridge and native PyTorch agreed
exactly with both ONNX variants on all 20 reused action cases (route, selected tool, grounded
arguments, and normalized action).

This closes an export-contract gap, not the capability gap.  The reused diagnostic suite has only
`0/19` tool-required exact actions and `0%` tool-selection accuracy, and no hardware WebGPU timing,
official benchmark score, browser environment, or real account was executed.  The bundle is
therefore suitable for a controlled demo artifact, but it is not a publication-quality browser-agent
result until the native and hardware gates in the workshop checklist pass.

### Grounded DOM pointer continuation and WebGPU check (m33)

The m32 result exposed a specific training-contract error: Mind2Web supplies positive/negative DOM
Candidates and backend node IDs, but the text-first normalized rows did not carry that observation
into the model context.  [`export_mind2web_grounded_rows.py`](../scripts/export_mind2web_grounded_rows.py)
now emits a bounded, deterministic 12-candidate snapshot before each action, preserving the public
revision, source-record split, and slot hashes.  The pointer head now supports a backward-compatible
browser vocabulary (`target_id`, `value`) without changing legacy checkpoint shapes; public browser
names are mapped only for auxiliary 51-way head supervision (`web_click` → `click`, etc.).

 A 32-step continuation warm-started the m31 10.5M model and retained its route/dense heads.  On six
held-out public-train decisions, exact pointer spans reached `3/6` (`50%`), but the fresh four-case
explicit-Metal-3 WebGPU action suite remained `1/4` exact (`25%`): tool selection was `3/3`, the
cancellation abstention was `1/1`, and all three web arguments copied a long candidate span starting
at `target_id=2932` instead of the gold `target_id=170`.  Schema validity was `4/4`; harness TTFA was
`18.65 ms` p50 and `19.385 ms` p95, with every case under 100 ms.  The full hash-bound receipt is
[`m33`](paper/results/raw/m33-mind2web-grounded-dom-webgpu-v1.json).

This is the desired realistic failure signal, not a success claim: compact DOM text alone does not
yet teach a 10.5M model to rank the correct candidate under a held-out website.  The next required
experiment is larger official-train coverage plus candidate-ranking/state-transition supervision,
followed by native browser execution; the four-case train-derived suite must not be promoted to an
official Mind2Web score.

### Gold-independent DOM candidate safety adapter (m34)

The m33 failure was specific: the model selected `web_click` correctly but its pointer span crossed
several serialized candidates and returned the wrong backend node.  The WebGPU runtime now has an
explicit safety adapter that ranks only the observed candidate attributes against task-token
overlap.  It selects the highest-scoring observed `target_id`; if no candidate matches, it preserves
the learned pointer output for unlabeled/icon-only controls.  The adapter does not read positive or
negative labels and is therefore a deployment policy, not a hidden test oracle.

On the unchanged four-case public-train held-out suite, the adapter improved complete structured
actions from `1/4` to `4/4`: three candidate-ranked web clicks and one cancellation abstention.
The explicit in-app WebGPU adapter was Apple Metal-3 (non-fallback), with p50 harness TTFA
`37.20 ms` and p95 `38.65 ms`.  The underlying learned pointer metric remains `3/6` exact spans;
the 4/4 result must not be described as pointer-head-only generalization or official Mind2Web
browser success.  See [`m34`](paper/results/raw/m34-mind2web-grounded-dom-webgpu-candidate-safety-v1.json).

### Native WebGPU action capability receipt (m40)

The [`webgpu-capability.html`](../spaces/localagent-webgpu/webgpu-capability.html) harness loads
the hash-bound 10,524,544-parameter action graph with an exact WebGPU provider request, queries
`navigator.gpu.requestAdapter()`, and records the exposed Apple Metal-3 adapter identity
(`vendor=apple; architecture=metal-3`). The expanded receipt covers three realistic local
dispatch cases—email, browser navigation, and a Notion write—with three warmups and 30 measured
repetitions per case. All three selected the expected tool and schema-valid argument on all 30
repetitions. The action graph's p50 input-throughput estimate was `1,355.9 tokens/s`, p50
end-to-end dispatch latency was `7.9 ms`, and the conservative graph-plus-host-tensor memory
estimate was `20.46 MB`.

This is now sufficient for the workshop gate's native WebGPU capability/latency requirement, but
it is not an external side-effect test: no email account, browser navigation, or Notion account
was touched, so `closed_loop_success` is explicitly `0`. The memory figure is an allocation
estimate because WebGPU/ORT exposes no driver VRAM counter. The two-case predecessor remains
[`m39`](paper/results/raw/m39-webgpu-native-capability-v1.json); the full three-case hash-bound
receipt is [`m40`](paper/results/raw/m40-webgpu-native-capability-notion-v1.json).

### Offline normalized mobile action protocol

[`src/localagent/eval/mobile.py`](../src/localagent/eval/mobile.py) now provides the common
prediction contract for AndroidControl and AITW rows after normalization. It extracts the
expected `mobile_*` calls from `localagent_v1`, accepts either `{name, arguments}` or
`{tool, args}` predictions, and reports tool accuracy, exact action accuracy, trajectory exactness,
optional coordinate proximity, and optional evaluator-provided target-box grounding. It validates
finite coordinates and malformed streams before scoring. This is an offline action diagnostic; it
does not simulate Android state or substitute for AndroidWorld's emulator reward. The emulator
runner should reuse this per-step contract while publishing device reward separately.

The scorer was replayed against the bounded normalized public training slices: 16 AndroidControl
records and 10 AITW records (130 tool calls total). Ground-truth replay is exact for both sources;
the sanity receipt is [`m22-mobile-action-score-replay-v1.json`](paper/results/raw/m22-mobile-action-score-replay-v1.json).
This validates the interchange and scorer only, not model quality.

### AndroidWorld emulator-result bridge

The official [AndroidWorld checkpointer](https://github.com/google-research/android_world/blob/main/android_world/checkpointer.py)
writes one `task_template_instance_id.pkl.gz` file per completed task instance under a timestamped
`run_...` directory. The upstream episode record contains the goal, task template, instance ID,
binary `is_successful` reward, episode length, runtime, exception information, and the full
step-level episode payload. [`src/localagent/eval/androidworld.py`](../src/localagent/eval/androidworld.py)
consumes that contract without importing AndroidWorld or starting `adb`: it hashes every result
file, rejects symlinks and malformed/duplicate instances, and reports per-task and overall
success rates. Supplying the exact expected task list and `n_task_combinations` is required for
`completeness.verified`; otherwise the receipt is intentionally marked `incomplete`.

Because upstream uses Python pickle, the parser's default loader accepts builtin-only fixtures.
Loading a full trusted emulator checkpoint requires the explicit
`--allow-unsafe-pickle` acknowledgement in [`scripts/aggregate_androidworld.py`](../scripts/aggregate_androidworld.py).
This is a local result-protocol bridge, not a live runner or an official AndroidWorld leaderboard
score. No AndroidWorld device result is published yet: the current environment lacks the
upstream emulator/`adb` prerequisites.

### AndroidWorld public task inventory (m51)

The public [`android_world/task_metadata.json`](https://github.com/google-research/android_world/blob/3e50888527ef9f29b9157ecd537e408008bb1c85/android_world/task_metadata.json)
was acquired at commit `3e50888527ef9f29b9157ecd537e408008bb1c85` and summarized in the
[`m51 receipt`](paper/results/raw/m51-androidworld-metadata-v1.json).  It contains 116 unique
task templates: 61 easy, 36 medium, and 19 hard; the median optimal budget is six steps, 96
templates are tagged parameterized, 24 require screen reading, and eight span multiple apps.
The task families cover messaging, calendar, contacts, browser, files, notes, recipes, settings,
media, and information retrieval—closely matching the requested mobile/email/productivity surface.

This receipt is a metadata inventory, not an AndroidWorld score.  It did not import AndroidWorld,
install APKs, invoke `adb`, launch an emulator, read screenshots, or feed task templates to SFT/RL.
The upstream environment reward and generated task instances remain required for a native mobile
claim; the inventory instead gives the WebGPU adapter a pinned, auditable task taxonomy and step
budget for future emulator runs.

### EnterpriseOps-Gym public card/API inventory (m52)

The pinned [EnterpriseOps-Gym dataset card](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym)
and Hugging Face API metadata were acquired at revision `c8e538eae8a6205294f0a86675fefdc1fac408f6`.
The card/API profile is frozen in the [`m52 receipt`](paper/results/raw/m52-enterpriseopsgym-metadata-v1.json)
and records the source hashes, schema, split sizes, and tool-set modes.  It makes an important
source discrepancy explicit: the card's About paragraph says **1,150** expert-curated tasks,
while its domain table sums to **1,115**; the API's downloadable row counts are 649 (`oracle`) and
637 for each `plus_5_tools`, `plus_10_tools`, and `plus_15_tools` configuration.  The domain table
also reports 104 email tasks, 79 email tools, eight enterprise domains, 512 tools overall, and a
9.15-step average (maximum 34).

This is metadata-only evidence.  The profile did not download parquet rows, read task verifiers or
server configuration, start Docker/MCP services, execute a state transition, or add benchmark text
to training.  A native EnterpriseOps claim still requires the resettable containerized servers and
SQL final-state verifiers; the existing 67-row email retrieval receipt remains a name-only selector
diagnostic, not task success.

## What can be trained

Only the following public demonstrations are currently training candidates:

| Source | What it teaches | WebGPU representation | Policy |
| --- | --- | --- | --- |
| [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) | Human mobile tasks over accessibility trees, screenshots, goals, and JSON actions; 15,283 demonstrations over 833 apps | Goal + compact accessibility tree → normalized action JSON | Train only on the official training partition; keep app/task generalization holdouts separate |
| [Android in the Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild) | Large-scale gesture and multi-step mobile behavior (715k episodes) | Instruction + screen description → gesture/action JSON; screenshots remain native-only | Train only after verifying the official archive and attribution; never mix evaluation episodes |
| [xLAM Function Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Tool selection and canonical argument filling | Compact tool schema + request → canonical JSON tool call | Existing `xlam_v1` adapter; exact denylist protects external evaluations |
| [Mind2Web train](https://huggingface.co/datasets/osunlp/Mind2Web) | Grounded browser navigation and forms | Cleaned DOM/accessibility tree + task → CLICK/TYPE/SELECT | Existing `mind2web_v1` adapter; only official train files |

AndroidControl is particularly useful for a first realistic converter because its official format
contains high- and low-level instructions, accessibility trees, screenshots, and JSON actions.
The converter should discard pixels for the text-only model, preserve a deterministic compact tree,
and retain the original episode/app/task identifiers in provenance.  A visual WebGPU policy can be
added later as a separate model; it must not silently change the text model's input contract.

## What must stay evaluation-only

The following are important reality checks, not extra SFT rows:

- [AndroidWorld](https://github.com/google-research/android_world): 116 hand-crafted tasks in 20
  apps with dynamically generated variations and live-emulator rewards.
- [WebArena](https://github.com/web-arena-x/webarena): 812 self-hosted realistic web tasks.
- [BrowserGym/MiniWoB++](https://github.com/ServiceNow/BrowserGym): reproducible browser primitives
  for smoke tests, not a substitute for enterprise workflows.
- [OSWorld](https://github.com/xlang-ai/osworld) and [OSWorld-Human](https://github.com/xlang-ai/OSWorld-Human):
  desktop multi-application workflows and human trajectories; use a VM runner and do not claim an
  official leaderboard score from a local run.
- [AgentNet / OpenCUA](https://github.com/xlang-ai/OpenCUA): 22.6K human-annotated computer-use
  tasks across Windows, macOS, and Ubuntu, with an offline AgentNetBench trajectory evaluator.
  Its action-reduction and state-action matching protocol is especially relevant to a compact
  state-conditioned WebGPU policy; keep screenshots, task labels, and official OS/app holdouts out
  of training until the dataset terms and split hashes are verified.

- [AppWorld](https://github.com/StonyBrookNLP/appworld): protected task/app/API-specific bundles;
  never unpack those into plaintext training data.
- [tau-bench](https://github.com/sierra-research/tau-bench) / tau3-bench: interactive user-agent-tool
  conversations and pass^k stability.
- [Apple ToolSandbox](https://github.com/apple/ToolSandbox): state dependencies, user simulation,
  intermediate milestones, and final state checks.
- [Toolathlon](https://github.com/hkust-nlp/Toolathlon): 32 applications, 604 tools, 108 long-horizon
  tasks, including Notion and email-like productivity workflows.  The repository terms need to be
  checked before any redistribution, so it remains evaluation-only.
- [MCPMark Verified](https://github.com/eval-sys/mcpmark): pinned Notion, GitHub, filesystem, Postgres,
  and Playwright environments with deterministic verifiers.
- [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym): the pinned
  card reports about 1,150 expert-curated tasks in its About paragraph, while its domain table
  totals 1,115 (including 104 email tasks), across eight domains and 512 tools.  The API exposes
  649 rows in `oracle` and 637 rows in each distractor configuration; these are configuration
  views, not additional task claims.  The official card describes containerized execution with
  SQL state verifiers and reports Apache-2.0; tasks/verifiers stay evaluation-only to prevent
  benchmark memorization.  The pinned HF revision is `c8e538eae8a6205294f0a86675fefdc1fac408f6`.
  The card/API reconciliation is frozen in the [`m52 metadata receipt`](paper/results/raw/m52-enterpriseopsgym-metadata-v1.json).
  The card-level inventory (104 email tasks) is not the same as a downloadable config row count:
  the bounded oracle and `plus_15_tools` email configs used below expose 67 matching task IDs, so
  the retrieval receipt reports 67 and retains 104 only as inventory context.
- [MCP-Persona](https://github.com/wwh0411/MCP-Persona): 173 personalized tool-calling tasks that
  include Notion and email MCP servers.  Its repository terms must be checked before redistribution;
  keep the simulated account state and checkpoint labels evaluation-only.
- [TUA-Bench](https://github.com/facebookresearch/TUA-Bench): 120 execution-based terminal tasks,
  including document editing, email management, and live-web information seeking.  It is CC-BY-NC
  and explicitly a benchmark, so it is not a training corpus.

The repository now has a fail-closed `agentnet_v1` adapter in
[`src/localagent/data/agentnet.py`](../src/localagent/data/agentnet.py) and a bounded normalizer in
[`scripts/normalize_agentnet.py`](../scripts/normalize_agentnet.py). It accepts both the official
`steps`/`ground_truth_actions` sample shape and the Hugging Face `traj`/`value.code` JSONL shape,
parses only literal `pyautogui`/`computer` calls, and requires a textual observation. Coordinate
actions remain `agentnet_*` tools for offline scoring; they are intentionally not relabeled as the
text-grounded WebGPU `click` or `type_text` tools. The one-sample integration receipt is
[`m17-agentnet-offline-adapter-sample-v1.json`](paper/results/raw/m17-agentnet-offline-adapter-sample-v1.json).
[`src/localagent/eval/agentnet.py`](../src/localagent/eval/agentnet.py) now supplies a dependency-free
AgentNetBench-compatible proxy for coordinate, text, keyboard, scroll, termination, and action-count
scoring. A ground-truth replay of the official sample scores `1.0` across 12 actions, which is only
a scorer sanity check; the pinned receipt is
[`m18-agentnet-scoring-proxy-sample-v1.json`](paper/results/raw/m18-agentnet-scoring-proxy-sample-v1.json).

A pinned metadata-only snapshot of the public Hugging Face export is also profiled in
[`m19-agentnet-metadata-profile-v1.json`](paper/results/raw/m19-agentnet-metadata-profile-v1.json):
22,532 unique task rows across Windows (12,364), macOS/Darwin (5,168), and Ubuntu (5,000), with
2–131 steps and 13 recorded low-level action types. The source JSONL is 18,840,344 bytes with
SHA-256 `9bb101e8373cd8cd1316f29d53c938b378f96aae1f09776a32bcc27454a0184d`; no screenshots or
trajectory payloads were consumed. This is an inventory receipt, not training data or an
AgentNetBench score.

The same boundary is now applied to a pinned [Toolathlon-GYM](https://github.com/eigent-ai/toolathlon_gym)
checkout. The configuration-only profile in
[`m20-toolathlon-gym-config-profile-v1.json`](paper/results/raw/m20-toolathlon-gym-config-profile-v1.json)
covers all 503 finalpool tasks and 25 MCP servers without opening task prompts, workspaces,
preprocessors, evaluators, or ground-truth outputs. It records 258 email tasks, 118 Notion tasks,
63 Playwright/browser tasks, 162 calendar tasks, and 419 filesystem tasks; the most common
multi-server pair is `emails+filesystem` (193 tasks). This gives the WebGPU tool bridge a concrete
long-horizon productivity gate while keeping the benchmark evaluation-only.

The pinned [MCPMark](https://github.com/eval-sys/mcpmark) metadata profile at commit
`cd45b7f57923b9b3985467f5139927575f83141c` is recorded in
[`m21-mcpmark-metadata-profile-v1.json`](paper/results/raw/m21-mcpmark-metadata-profile-v1.json).
The checkout contains 239 metadata rows: 169 `standard` L3 tasks and 70 `easy` L1 smoke tasks,
covering 38 Notion, 35 browser/Playwright, 40 filesystem, 33 GitHub, and 93 database tasks.
State types are text (133), URL (71), and video (33); the profile retains neither descriptions nor
state assets. MCPMark's own runner supports repeated runs and pass@k/pass^k aggregation, so it is
the next appropriate stability gate once the local MCP/browser dependencies are installed.

The current upstream [tau2-bench](https://github.com/sierra-research/tau2-bench) repository is the
maintenance line for the original tau-bench paper and now points new work to tau3-bench.  Its
base split must be used for a complete evaluation, and the leaderboard protocol requires repeated
trials for Pass^k; a single trajectory replay or a historical tau-bench number is not a current
deployment score.

[`src/localagent/eval/mcpmark.py`](../src/localagent/eval/mcpmark.py) now provides a fail-closed
local result aggregator: it requires every expected task for every run before reporting
pass@1/pass@k/pass^k, mean turns, latency, and token use. No result has been produced here because
Docker/MCP services are not installed; the implementation is a protocol bridge, not a score.

### MCPMark task-router proxy (m37)

To quantify the model's current productivity/tool gap without pretending to execute external
accounts, [`eval_mcpmark_task_router.py`](../scripts/eval_mcpmark_task_router.py) consumes only the
public `meta.json` and `description.md` files from the pinned MCPMark checkout. It evaluates both
the 169-task standard suite and the 70-task easy suite against service-level tool families:
Notion, Playwright/browser, filesystem, GitHub, and Postgres. Task text is used transiently for
feature extraction and is not retained in the receipt; verifiers and MCP servers are never run.

The frozen 62-tool AndroidControl dispatch child achieved only `15.90%` combined route accuracy,
`0%` selector top-1, and `0.42%` selector top-3 on 239 tasks. This is a useful transfer failure:
mobile action adaptation does not transfer to stateful Notion/email/browser/database workflows.
The result is a service-routing proxy, not MCPMark task success, pass@k, or leaderboard evidence;
the complete source and checkpoint hashes are in [`m37`](paper/results/raw/m37-mcpmark-task-router-proxy-v1.json).

### MCP service-contract transfer probe (m38)

The first routing failure is now separated from the language-model representation. The
[`train_mcp_service_probe.py`](../scripts/train_mcp_service_probe.py) experiment creates 28 short
service/tool-contract examples for filesystem, GitHub, Notion, Playwright, and PostgreSQL. It
trains only the five-way route head and the description-based dense selector; the 10.5M language
model backbone is copied unchanged (aggregate backbone delta `0.0`). No MCPMark description,
state fixture, verifier, or server is used as training input.

On the same untouched 169-task standard and 70-task easy MCPMark descriptions, the child reaches
`46.86%` combined route accuracy, `37.24%` selector top-1, and `69.87%` selector top-3, compared
with m37's `15.90%`, `0%`, and `0.42%`. Standard is `50.89% / 37.87% / 66.27%`; easy is
`37.14% / 35.71% / 78.57%`. This is a meaningful head-transfer result, not an execution score:
the official MCPMark runner, stateful MCP servers, and verifiers are still required before any
claim of task success. The complete hash-bound receipt is
[`m38`](paper/results/raw/m38-mcp-service-contract-probe-v1.json).

### MCP service-contract matched no-transfer control (m53)

To test whether the m38 selector result came from the pretrained representation rather than only
head fitting, the same 28 generated service/tool contracts, tokenizer, architecture, 800 route
updates, 800 selector updates, and seed were run with a fresh random backbone.  The control was
evaluated against the same 239 public MCPMark task descriptions; task text was transient, and no
MCP server or verifier was executed.  The [`m53 receipt`](paper/results/raw/m53-mcp-service-contract-no-transfer-v1.json)
records the child hash and self-hash.

| Condition | Route accuracy | Selector top-1 | Selector top-3 |
| --- | ---: | ---: | ---: |
| Pretrained backbone frozen (m38) | 46.86% | 37.24% | 69.87% |
| Fresh random backbone (m53) | 47.28% | 23.43% | 64.44% |
| Random − pretrained | +0.42 pp | −13.81 pp | −5.44 pp |

This is bounded transfer evidence: the pretrained representation materially improves tool-family
selection on unseen Notion/browser/database descriptions, while route classification is unchanged.
It is not MCPMark task success, pass@k, verifier success, or evidence that live accounts can be
controlled; the official runner remains a workshop-gate requirement.

### MCPMark current-checkpoint routing proxy (m60)

The same untouched pinned MCPMark descriptions were rerun with the m56 stateful child and the m59
ToolSandbox-SFT child.  The [`m60 receipt`](paper/results/raw/m60-mcpmark-current-checkpoint-router-proxy-v1.json)
binds all 169 standard and 70 easy rows, both checkpoint hashes, and the source manifests.

| Child | Standard route / selector top-1 / top-3 | Easy route / selector top-1 / top-3 |
| --- | --- | --- |
| m56 | 17.75% / 3.55% / 13.61% | 12.86% / 2.86% / 12.86% |
| m59 | 20.71% / 2.37% / 14.79% | 22.86% / 1.43% / 14.29% |

ToolSandbox SFT therefore transfers some service-family routing but does not improve exact MCP
tool selection on this untouched proxy.  The task descriptions, state fixtures, servers, and
verifiers remain evaluation-only; this is not MCPMark task success or a leaderboard result.  The
next meaningful training input must be a permitted schema-conditioned MCP corpus, followed by the
official stateful server/verifier run.

### ToolSandbox public scenario metadata profile (m54)

The pinned [Apple ToolSandbox](https://github.com/apple/ToolSandbox) source is now inventory-bound
at commit `165848b9a78cead7ca7fe7c89c688b58e6501219`.  The dependency-free
[`profile_toolsandbox_metadata.py`](../scripts/profile_toolsandbox_metadata.py) parses only the
four public scenario modules with Python's AST; it does not import ToolSandbox or retain user
prompts.  The receipt reports 129 unique base scenarios: 19 single-tool, 54 multiple-tool, 28
multiple-user-turn, and 28 insufficient-information scenarios.  Literal source metadata names 32
tools, 59 canonicalization tags, and 24 state-dependency tags.  The upstream `named_scenarios`
augmentation policy expands each base definition into four distraction levels plus four schema
scramble variants, or 1,032 source-level variants before runtime tool-similarity ranking.

This is a public-source inventory, not a model score or training set.  The Apple Software License
and `ACKNOWLEDGEMENTS` are preserved in the source manifest and require a separate redistribution
review.  Native evaluation still requires the pinned simulator, state databases, user simulator,
milestone verifiers, and complete upstream result coverage; no external API, tool, or account was
used here.  The complete hash-bound receipt is
[`m54`](paper/results/raw/m54-toolsandbox-metadata-v1.json).

The public [Apple ToolSandbox](https://github.com/apple/ToolSandbox) is the next stateful
productivity gate. Its scenarios model implicit tool dependencies, user simulation, messaging,
canonicalization, distraction tools, and insufficient-information abstention; the upstream CLI
writes one JSON result record per scenario with milestone similarity, minefield similarity,
turn count, exceptions, and a milestone-to-turn mapping. The dependency-free
[`src/localagent/eval/toolsandbox.py`](../src/localagent/eval/toolsandbox.py) bridge consumes that
official `result_summary.json` shape, hashes the source, mirrors the upstream category rule for
distraction augmentations, and fails closed unless the expected scenario list is present.
[`scripts/aggregate_toolsandbox.py`](../scripts/aggregate_toolsandbox.py) writes a receipt without
executing tools or importing the ToolSandbox package. This is a local result-summary aggregation
bridge; a live ToolSandbox run and any score remain pending.

The official [BrowserGym environment](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/env.py)
uses the Gymnasium episode contract: each browser action is passed to `step(action)` and returns
observation, reward, terminated, truncated, and info, with task-specific validation supplying the
reward. [`src/localagent/eval/browsergym.py`](../src/localagent/eval/browsergym.py) defines the
portable JSONL projection of that contract (`task_id`, `seed`, ordered action/reward steps, and
terminal flags), hashes the log, checks exact task/seed coverage, and reports per-task reward,
success, step count, and action errors. [`scripts/aggregate_browsergym.py`](../scripts/aggregate_browsergym.py)
consumes the projection without Playwright or a browser. It is a local BrowserGym/WebArena/
WorkArena/MiniWoB protocol bridge; no live browser score is claimed until the pinned runtime is
installed and the runner produces a complete receipt.

### AgentNet/OpenCUA desktop-action result bridge

The public [AgentNet dataset](https://huggingface.co/datasets/xlangai/AgentNet) contains
human-annotated desktop computer-use trajectories across Windows, macOS, and Ubuntu. The
[OpenCUA repository](https://github.com/xlang-ai/OpenCUA) publishes AgentNetBench as an offline
low-level action evaluator. Its coordinate, text, keyboard, scroll, and termination signals are
useful for a WebGPU policy, but the source trajectories are screenshot-grounded; a text-only
checkpoint must not convert them into training rows without an accessibility/vision bridge.

The existing [`src/localagent/eval/agentnet.py`](../src/localagent/eval/agentnet.py) mirrors the
per-trajectory action protocol without opening screenshots. The new
[`src/localagent/eval/agentnet_results.py`](../src/localagent/eval/agentnet_results.py) joins a
ground-truth JSONL export with a prediction JSONL stream, rejects duplicate or mismatched task
IDs, hashes both inputs, and reports mean action score, exact trajectory rate, action-count
penalties, action-type scores, and platform breakdowns. The
[`scripts/aggregate_agentnet.py`](../scripts/aggregate_agentnet.py) CLI requires an explicit
expected-ID list for a complete receipt. This remains an offline AgentNetBench-compatible proxy,
not a native AgentNetBench leaderboard score or an OS/VM run.

### tau2-bench interactive tool-result bridge

The public [tau2-bench](https://github.com/sierra-research/tau2-bench) runner evaluates a
half-duplex user simulator and agent against stateful airline, retail, telecom, and
banking-knowledge tool environments. Its official [task-schema/evaluation guide](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)
is important for interpretation: the default reward is an outcome product (database state plus
required communication), while `evaluation_criteria.actions` is usually one reference
trajectory, not a required script. The official [leaderboard guide](https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md)
requires the complete base split, consistent configuration, and at least four trials for a
publication-quality Pass^k submission.

[`src/localagent/eval/tau2.py`](../src/localagent/eval/tau2.py) consumes the upstream
`Results` JSON contract in both supported storage forms: one monolithic `simulations[]` file,
or `results.json` plus individual `simulations/*.json` files. It validates task/trial IDs,
reward, termination, duration, and metadata; hashes every source file; excludes
`infrastructure_error` runs as the upstream metrics do; and computes the upstream combinatorial
Pass^k estimator per task (not a misleading all-trials product). The
[`scripts/aggregate_tau2.py`](../scripts/aggregate_tau2.py) CLI accepts expected cases in the
stable `domain/task_id@trial` form and marks a receipt incomplete until exact coverage and the
requested trial count are present. This is a dependency-free result bridge: it does not invoke
tau2, user simulation, or any email/retail tool, and no tau2 score has been produced in this
environment.

The catalog also binds the newer realistic environments that should be part of the workshop-grade
evaluation matrix:

- [MobileGym](https://github.com/Purewhiter/mobilegym): 28 simulated mobile apps and 416 reusable
  task templates with deterministic judges.  The code is Apache-2.0, while the released task/data
  terms are CC-BY-NC-4.0, so it is evaluation-only here.
- [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld): 201 long-horizon tasks across 20 mobile
  apps, including agent-user interaction and MCP-augmented workflows.
- [MemGUI-Bench](https://github.com/lgy0404/MemGUI-Bench): 128 GUI tasks (with a 40-task quick
  subset) that test interaction memory across browser, office, and file workflows.
- [WorkArena](https://github.com/ServiceNow/WorkArena): enterprise ServiceNow browser tasks,
  integrated through [BrowserGym](https://github.com/ServiceNow/BrowserGym); upstream task terms
  require review before redistribution.
- [WebLINX](https://github.com/mcgill-nlp/weblinx): roughly 100k grounded browser interactions
  across 2,300 demonstrations with dialogue context.  The official dataset terms remain binding.
- [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2): the dated 2026-06-24 release of the
  long-horizon desktop benchmark; gated task files/assets remain outside training and public
  checkpoints.
- [Toolathlon-GYM](https://github.com/eigent-ai/toolathlon_gym): 503 tasks over 25 local MCP servers,
  including reproducible email and Notion-style workflows without live external accounts.
- [GUIOdyssey](https://github.com/OpenGVLab/GUI-Odyssey): 8,834 cross-app mobile episodes over
  212 apps and six devices; the latest public snapshot provides explicit app/device/task splits.
- [MobileAgentBench](https://github.com/MobileAgentBench/mobile-agent-bench): 100 Android tasks
  over 10 open-source apps, designed as a small reproducible emulator gate before larger suites.
- [AgentBench Function Calling](https://github.com/THUDM/AgentBench): the current containerized
  release covers five interactive environments; the original benchmark spans eight environments.
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena): 910 visually grounded WebArena
  tasks plus released human trajectories; a text-only run is explicitly a modality ablation.
- [OmniACT](https://huggingface.co/datasets/Writer/omniact): 9,799 desktop/web instruction-image
  pairs with PyAutoGUI scripts; it is a vision benchmark, not a text-first score.
- [WorldGUI](https://showlab.github.io/WorldGUI/): 315 dynamic desktop tasks over 10 applications
  with varied initial states and pre-action sequences.
- [macOSWorld](https://github.com/showlab/macosworld): 202 multilingual tasks over 30 macOS
  applications, including a safety subset.
- [ASSISTGUI](https://github.com/showlab/assistgui): 100 Windows productivity tasks over nine
  applications with project-file state.

The existing BFCL, WebLINX, and protected Mind2Web/browser captures in `data/private/` remain
evaluation-only under the same rule.  A public license does not make a benchmark safe to train on:
task prompts, verifier code, and gold state can directly inflate the score.

### Capability-to-benchmark map

| Capability | Representative public evaluations | What must be measured | Current WebGPU bridge |
| --- | --- | --- | --- |
| Mobile single-step and long-horizon control | AndroidWorld, MobileGym, MobileWorld, GUIOdyssey, MobileAgentBench, PhoneWorld | grounded action validity, milestone reward, episode success, app/device generalization | accessibility-tree text plus goal; no screenshot grounding yet |
| Browser navigation and visual grounding | BrowserGym/MiniWoB++, WebArena, WorkArena, WebLINX, VisualWebArena | element accuracy, operation F1, DOM/state transitions, task success | cleaned DOM/A11y text; visual suites are modality ablations |
| Desktop/computer use | AgentNet/AgentNetBench, OSWorld/OSWorld-V2, MemGUI-Bench, WorldGUI, macOSWorld, ASSISTGUI | VM task success, offline action/trajectory accuracy, long-horizon recovery, latency, safety | AgentNet offline parser is ready; coordinate-to-vision bridge and VM runners are pending |
| Stateful tools and productivity | Toolathlon-GYM, MCPMark, EnterpriseOpsGym, AgentBench FC, AppWorld, tau-bench | exact calls, schema validity, state delta, pass^k, abstention | local email/Notion mocks and retrieval sidecar; no external accounts |

This map is intentionally asymmetric: a text-first 10.5M model can produce a credible structured
tool/DOM bridge, but it cannot claim screenshot-level grounding or native OS control.  Those are
separate acceptance gates, not hidden quality assumptions.

## Pipeline for a WebGPU model

1. **Acquire and bind.** Download only a catalog row with `train_policy: train`; record the source
   URL, revision, license evidence, bytes, and SHA-256.  Run
   `python scripts/validate_realistic_agent_catalog.py` before ingestion.
2. **Normalize.** Convert AndroidControl/AITW records into the canonical `Conversation` schema.  Use
   an explicit decoder-produced intermediate row and
   [`normalize_realistic_agent_jsonl.py`](../scripts/normalize_realistic_agent_jsonl.py), which
   emits the existing `localagent_v1` format.  The normalizer rejects screenshot-only rows and
   unknown action arguments, so a converter that cannot prove the official split or text projection
   fails closed.
3. **Pre-train and mid-train.** Keep broad text/code pretraining separate from the UI/tool mixture;
   then mix verified mobile/browser/tool demonstrations with a fixed, documented ratio.  Preserve
   disjoint app/task/domain slots and the existing exact prompt denylists.
4. **SFT.** Train canonical tool calls and browser actions on the training candidates plus
   operator-reviewed synthetic examples.  Generate email/Notion-like workflows against local mock
   tools; do not copy Toolathlon, MCPMark, AppWorld, or EnterpriseOpsGym task text.
5. **RL simulation.** Use local stateful mocks with checkpoint rewards: tool selection, argument
   validity, state delta, user-facing completion, and safe abstention.  Re-run the same seed suite
   for pass@1/pass^k and keep the external benchmark tasks untouched.
6. **Evaluation.** Report capability by family, not one blended number: mobile action/episode
   success, browser element/task success, desktop task success, tool-call exact match, state-delta
   success, abstention, TTFA, tokens/s, and peak WebGPU memory.  Every score must name the exact
   upstream revision and whether it came from a local runner or an official leaderboard protocol.

## WebGPU observation contract

The first deployable bridge should use accessibility/DOM text and compact state deltas.  This keeps
the model under the WebGPU token and memory budget while still testing realistic control.  Screenshots
are retained in provenance for AndroidControl/AITW but are not fed to the text-only checkpoint.
Native screenshots and pixel gestures belong to a future multimodal branch, with a separate model
config and evaluation receipt.

## Reproducible AndroidControl bridge pilot

One bounded acquisition has now been exercised end to end without placing source payloads in Git:

- Official object: `gresearch/android_control/android_control-00000-of-00020`, generation
  `1717537205165914`, 2,492,987,829 bytes, upstream MD5 `QuhP+X5aiP+iD5amlNM6tA==`.
- Local range: bytes `0..33,554,431`, SHA-256
  `380b58842079ae52cf7b2e92e74940ec1c571c3eb3955785a2d8556ebb4e17cd`; the range is explicitly
  marked as a truncated-GZIP acquisition and contains complete records through the selected bound.
- Split binding: official `splits.json`, SHA-256
  `aa0943b0b4eb354796f3daf4a6e36836918c71254db2d4060b45ae98213ad2b6`, with only `train` episode
  IDs selected.  Eight records (episode IDs `0,20,40,60,80,100,160,220`) were normalized to 16
  deterministic Conversation rows after level-1 enrichment.
- Normalized JSONL: 1,511,467 bytes, SHA-256
  `dce5c7738fc3702f16ad5dabddbcafa8f67edad2fe1ad50d937df4dc7456e7f7`.  The decoder projects
  the official accessibility protobuf to bounded text and rejects screenshot-only records.

A second bounded train source now exercises the larger [AITW release](https://github.com/google-research/google-research/tree/master/android_in_the_wild):

- Official object: `gresearch/android-in-the-wild/general/general-00001-of-00321`, generation
  `1686095863743117`, 3,249,449 bytes, upstream MD5 `o86d8TEjui+XzNjKrxywxA==`.
- Official `standard.json` split binding is SHA-256
  `324ca94dbbb0778cee7fd1f2bd8dcba20cce592fbef6aec41ee475ed47ae7681`; all five complete
  episodes in this shard are in the official `train` list.
- The AITW adapter reconstructs step-level TFRecords into complete episodes, converts normalized
  dual-point gestures to pixel-space click/swipe actions, preserves type/home/back/enter actions,
  and projects UI annotation text/bounds without copying screenshots.  Five episodes became 10
  enriched Conversation rows (canonical JSONL SHA-256
  `31bdd0320ddcd75345aca73fd7ca81c05d11b6831091da94faca57b841df5ce6`).

The v2 projection used for the corrected continuation keeps each step instruction after the
observation: AndroidControl normalized JSONL SHA-256
`705bf99465b5feb2b24f03e5c81cc2c1560ac8f28200eb83ed8c921626c995fa` (16 rows) and AITW normalized
JSONL SHA-256 `1f0a08fac5684f21d62a10855fc05171b4d7ba699f8aae51f062d01f7b52031` (10 rows). The
source manifests remain bound to the same official train-only episode selections.

An earlier 8-step continuation over the 26-row AndroidControl+AITW mixture reported mean
normalized-row loss `8.6239` → `7.9730`, but it loaded the default byte tokenizer against the
16K-BPE parent checkpoint.  That receipt is retained only as a historical invalidation example;
its token accuracy and checkpoint must not be used for model-selection claims.

The corrected BPE-lineage continuation uses the tokenizer recorded in the parent checkpoint and
the same train-only rows.  It reduced mean loss `5.4868` → `4.7944` and assistant-token accuracy
rose `25.68%` → `31.69%`; exact trajectory accuracy remained `0/26`.  The child checkpoint is
kept outside Git (`cc12da27808d251df46d1beb649622e896cb32233dc2b6317deb917dec93c08a`) and is a
language-model bridge measurement only, not evidence of emulator success.  The tokenizer
compatibility gate is now covered by `tests/test_train_androidcontrol_pilot.py` and the pilot
refuses a missing or vocabulary-mismatched BPE artifact.

### Public AndroidControl 84K text-action continuation (m35)

The public [OfficerChul/Android-Control-84k](https://huggingface.co/datasets/OfficerChul/Android-Control-84k)
mirror exposes 82,944 train rows and a deliberately balanced 904-row test set derived from
Google's Apache-2.0 AndroidControl release.  [`ingest_androidcontrol_json.py`](../scripts/ingest_androidcontrol_json.py)
converts the LLaMA-Factory JSON shape into the canonical `Conversation` schema and records both
raw-file and normalized-output hashes.  The train/test files are never mixed.

This first scale-up is intentionally text-action only: the mirror supplies screenshot paths, but
the current checkpoint has no vision encoder, so the adapter marks every row
`visual_input_omitted=true` and `grounding_evaluable=false`.  A deterministic stratified 4,096-row
train slice (all 42 long-press examples and balanced coverage of the other train actions) was
continued for 32 BPE SFT steps from the 10.52M parent.  On all 904 held-out test rows, teacher-forced
assistant-token accuracy rose `59.28% → 65.47%` and mean loss fell `2.789 → 2.049`; teacher-forced
exact assistant sequences stayed `0/904`.

A second 300-step frozen-feature probe kept the adapted backbone fixed and trained only route and
dense dispatch heads over the 62-tool pool.  Held-out route accuracy was `100%`, while selector
top-1 was `46.24%` overall: navigate-back/open-app `74.4%`, scroll `66.4%`, wait `51.2%`, input
text `48.0%`, click `20.0%`, and long-press/home `0%`.  The transfer audit found matching config,
shapes, and tokenizer (`51` shared tensors); LM continuation moved embedding/attention/FFN norms
by `0.52%/0.22%/0.28%/0.01%` relative L2, while the frozen-head probe moved only dispatch heads
(`107.3%` relative L2).  This supports freezing the transferred backbone before dispatch-head
specialization, but it is not evidence that transfer improves emulator reward.

The complete source, split, training, selector, and weight-audit identities are in the
[`m35 receipt`](paper/results/raw/m35-androidcontrol-84k-text-action-transfer-v1.json).  Because
pixels were omitted, m35 must not be reported as AndroidControl visual grounding, AndroidWorld
success, or WebGPU hardware throughput; a workshop claim still requires a vision-capable adapter
and a native emulator run on the official evaluation split.

### AndroidControl dispatch balance ablation (m36)

The m35 split exposes a measurable train/test support gap: the 4,096-row train slice contains only
42 `mobile_long_press` rows and no `mobile_navigate_home` rows, while the public test contains 125
long-press and 29 home rows.  The dispatch runner now supports an explicit, hash-bound
`--focus-tool`/`--focus-repeat` ablation that adds synthetic and state-conditioned views without
touching the 904-row evaluation file.

Oversampling both missing/rare actions by 32 repeats did not solve the gap: held-out selector
top-1 fell from `46.24%` to `33.63%`, while both long-press and home remained `0%`.  The route head
stayed at `100%`, and the frozen backbone did not move.  This negative result is retained because
it rules out a tempting but overfit weighting fix; the next valid intervention is a licensed train
split containing those actions or a separately hash-bound real-device continuation.  The complete
comparison and head-movement audit are in the [`m36 receipt`](paper/results/raw/m36-androidcontrol-dispatch-balance-ablation-v1.json).

### Browser runtime receipt (local compatibility smoke)

The exported fp32 bundle was then exercised in the existing browser harness with an explicit
single-provider WebGPU session.  This is a provider/dispatch receipt, not a benchmark score for
AndroidWorld, WebArena, OSWorld, MCPMark, email, or Notion: the suite has eight one-step local DOM
fixtures, text-only observations, semantic targets, no screenshots, no physical cursor, and no
external navigation.

- Bundle: `sft_realistic_mobile_pilot`, 10,524,544 parameters; checkpoint SHA-256
  `268cd21d8f6a49e4e63c001ef73a26c67820d33407e5bb611a03382966700d1f`; bundle-manifest SHA-256
  `57cedfe061174fddf44cdae1d280bf1da9baef286f82655fdf17a78f5cbf139c`.
- Requested provider: `webgpu` with one ORT Web session; the observed adapter was Chrome
  SwiftShader (`google`), so this run demonstrates software WebGPU compatibility rather than a
  hardware-GPU throughput claim.  The model artifact was fetched, size-checked, and SHA-256
  verified in-browser before session creation.
- Result: 8/8 independent action schemas valid; exact action 6/8 (75%); final DOM/state
  transition 6/8 (75%); closed-loop success 6/8 (75%).  Click, double-click, type, scroll, drag,
  and cursor-move cases passed; the single key-press and local-navigation cases failed.
- Timing: harness TTFA p50 `157.6 ms`, closed-loop p50 `165.6 ms`, tool dispatch p50 `0.35 ms`.
  The slowest drag/scroll cases make the p90 closed-loop latency `4.28 s`; only 3/8 cases both
  completed and succeeded within a 250 ms deadline.  Treat p50 and deadline attainment separately.
- Versioned receipt JSON: [`m6-webgpu-realistic-mobile-mixture-browser-result.json`](paper/results/raw/m6-webgpu-realistic-mobile-mixture-browser-result.json), SHA-256
  `7b3872cd6e153b340b2fe9bc0a47d4b339635264c7ae7f57f0b2aeb21ffb6c26`.

The same eight-fixture control run under WASM (fp16 bundle) completed at roughly `33.3 ms` p50
with the same 75% exact/final-DOM/end-to-end rate.  Explicit WebGPU with that fp16 artifact did
not create a usable session because the selected device rejected `f16` gather kernels; this is a
runtime capability failure, not a model-quality score.  The fp32 export is therefore the current
compatibility fallback for WebGPU devices without shader-f16 support.  Neither receipt is evidence
that the model can yet control a real Android emulator, browser account, email system, Notion
workspace, or MCP server; those evaluations remain required before publication.

### Text-first mobile/productivity closed-loop pilot (custom 60-tool bundle)

The corrected mobile projection keeps the action instruction at the end of each observation, so a
left-truncated WebGPU context cannot discard the command after a long accessibility-tree dump. A
separate additive dispatch bundle was exported from the 10.5M-parameter mobile-dispatch child:

- Parent checkpoint: `sft-realistic-mobile-mixture-pilot.pt`, SHA-256
  `268cd21d8f6a49e4e63c001ef73a26c67820d33407e5bb611a03382966700d1f`.
- Child checkpoint: `sft-realistic-mobile-dispatch-pilot-sft200-v2.pt`, SHA-256
  `3606aa01952de1f006c039fef9f64f6184cc611ba320452674a71bbfc8ac137f`.
- Exported pool: 60 schemas (the stable standard 50 plus 10 `mobile_*` schemas), with fp32
  `model.onnx`/`action_model.onnx` parity max drift `9.86e-06`/`6.59e-06`.
- Bundle manifest SHA-256 `e6d56e433f7f6f9bcc712db241a21cf0cfbc1a3a36c388d2415182cfc721e427`;
  model and action graph SHA-256 values are pinned in the raw receipt.
- Local browser receipt: [`m7-webgpu-mobile-productivity-pilot.json`](paper/results/raw/m7-webgpu-mobile-productivity-pilot.json),
  SHA-256 `7887eaebd7512c360c4a99e9292e41634bce7f9f0e4899700a4ed7253015c507`.

The receipt runs nine deterministic steps against an in-memory Android/Gmail-style state plus
local Notion and email records. It uses explicit WebGPU on Chrome SwiftShader (software WebGPU),
text-only observations, no screenshots, no trusted OS input, and no external accounts. All nine
outputs were independently schema-valid. Mobile actions were 7/7 exact and closed-loop, but the
learned dense selector chose the wrong standard tool for both productivity tasks (0/2 exact), so
the aggregate is 7/9 (77.8%) and must not be presented as a real-device or real-account score.
The seven mobile successes use the explicitly reported `mobile_lexical_guard`; they are a runtime
contract check, not evidence that the dense selector has solved mobile dispatch. The held-out
selector result remains 40% top-1 on ten mobile rows, and the next acceptance gate is a no-guard
mobile ablation plus emulator/real-environment evaluation.

That 40% held-selector number is now historical only.  The pilot script had accidentally loaded the
default byte tokenizer while fine-tuning a 16K BPE checkpoint, so its selector metric was not a
valid learned-policy comparison.  The corrected BPE-tokenizer run below is the authoritative
dispatch measurement.

### Corrected BPE-tokenizer productivity dispatch and WebGPU receipt

The corrected continuation starts from the same 10.5M-parameter SFT parent but loads the tokenizer
recorded in the checkpoint (`data/tokenizer-webgpu-proxy-16k.json`) instead of the byte-level
default.  This removes the Python/browser feature mismatch that invalidated the earlier selector
diagnostic.

- Child checkpoint: `sft-realistic-mobile-dispatch-productivity-v6.pt`, SHA-256
  `be4f1216c88bfc6b12554e5d6324aa581e20409d6b213a3e47eaf5a3cc9f1583`.
- Training report: 3,000 steps; productivity train selector `12/12` (100%), productivity held
  selector `3/4` (75%), route accuracy `100%` on both splits.  The broader mobile held selector
  remains `4/10` (40%), so no-guard mobile control is not yet publication-ready.
- Export: 62 schemas, 10,524,544 parameters, fp32 ONNX parity max drift `8.58e-06` for logits and
  `6.14e-06` for hidden states.  Bundle-manifest SHA-256
  `5b711ff9768db66ef7a4d855bcb339244dc88ec1e6f6b5bf6eb5180b62f5644d`.
- Browser receipt: [`m9-webgpu-mobile-productivity-pilot-v6.json`](paper/results/raw/m9-webgpu-mobile-productivity-pilot-v6.json),
  SHA-256 `8fee312a92e897591c82f2327b6aa04693d412baa15ad14f03d234001bd1ad6c`.  On Chrome SwiftShader
  software WebGPU, the nine-step local state suite achieved schema validity `9/9`, exact tool
  selection `9/9`, exact arguments/action `8/9`, state transitions `8/9`, and closed-loop success
  `8/9`.  The seven mobile rows still use the explicit lexical guard; the dense selector passed
  the email row and missed the Notion title/content arguments.  Closed-loop latency ranged from
  `121 ms` to `7.44 s` with a `148 ms` median, so this is compatibility evidence rather than a
  100–300 tokens/s hardware-WebGPU claim.

The corrected receipt is therefore a reproducible local WebGPU gate, not evidence of AndroidWorld,
OSWorld-V2, WorkArena, WebLINX, MCPMark, Toolathlon-GYM, real email, or real Notion access.  Those
environment runners and official task splits remain required before a workshop or public model
claim.

### Retrieval sidecar ablation (v9r)

The v9r bundle adds an explicit 256-dimensional CRC32 character n-gram retrieval sidecar.  It is
not trained weights and is reported separately from the dense neural selector; its purpose is to
provide an auditable open-tool fallback for small WebGPU catalogs without reintroducing the
mobile lexical guard.  The sidecar uses clean synthetic action examples rather than inherited
accessibility dumps, and compacts a trailing `instruction:` field when present.

- Checkpoint: `sft-realistic-mobile-dispatch-productivity-v9r.pt`, SHA-256
  `523946f03fd8439e0c085ad9cbcab3bc43bcf3d28ef9385d9c42666a6f9eb0ec`.
- Bundle manifest SHA-256 `0785de8e1f84efb0ca68082bbb56e5b366ccf5212679f249091ed05cfc106af5`;
  dispatch sidecar SHA-256 `da09acd9df9e7073555543c99f83c579a52c3f4de4f20f67310877c0e26bdbba`.
- Receipt: [`m10-webgpu-mobile-productivity-v9r.json`](paper/results/raw/m10-webgpu-mobile-productivity-v9r.json),
  SHA-256 `9b69b0ff5e8b20f3756de4ece767883a1aa42a1820312e04400b5fca49973f3c`.
- With `mobile_guard=0&selector=retrieval`, Chrome SwiftShader WebGPU achieved 9/9 exact tools,
  9/9 exact arguments, 9/9 state transitions, and 9/9 closed-loop success across seven mobile
  actions plus email and Notion.  The same receipt's learned `dense_selector` no-guard ablation
  achieved 4/9 overall (2/7 mobile, 2/2 productivity).  This separation prevents a retrieval
  baseline from being misreported as neural tool-selection accuracy.

The retrieval sidecar is a deployment fallback, not a substitute for improving the learned
selector: the external mobile held-out neural selector is 6/10 (60%) after balanced synthetic
augmentation.  Workshop acceptance still requires a stronger no-guard neural result, an official
runtime benchmark, and hardware-WebGPU performance measurements.

### Stateful mobile/productivity trajectory gate (v11)

The next continuation adds 11 state-conditioned examples to the v10 parent-head probe while
keeping the 10.5M-parameter backbone frozen.  The local browser harness feeds the resulting state
JSON back into each next prompt and independently checks the bundle's `meta.json` schemas, exact
tool/argument equality, a preconditioned state transition, and complete-trajectory pass@1.  The
suite has three workflows and 13 ordered steps: Gmail compose/send (6), Notion capture (2), and a
local mail-page search/open flow (5).  It is deliberately text-first and uses no screenshots,
real accounts, MCP servers, or trusted OS input.

- Child checkpoint: `sft-realistic-mobile-dispatch-productivity-v11.pt`, SHA-256
  `4cccca1139699776c555876bb282783276f6e84b1a5e7e949dd83d5a2e8140ed`.
- The child keeps the v10 held-out selector results: mobile `6/10` (60%), productivity `3/4`
  (75%), and route accuracy `100%` on both; train selector is `86.21%` after 3,000 steps.
- The exported 62-tool bundle passes fp32/fp16 parity (fp32 logits drift `8.58e-06`; fp16
  logits argmax agreement `1.0`) and was requested through the in-app browser's WebGPU provider.
- The stateful run is a negative result: schema validity `13/13`, exact actions `0/13`, state
  transitions `0/13`, closed-loop success `0/13`, and complete-trajectory pass@1 `0/3`.  The first
  Gmail action emitted `mobile_input_text({text: "app"})`; the Notion and browser workflows both
  started with `mobile_open_app({app_name: "app"})`, so the validator failed closed before applying
  any state transition.

The complete identity, parity, browser condition, and first-failure records are in the
[`m15 receipt`](paper/results/raw/m15-webgpu-mobile-productivity-trajectories-v11.json).  This
diagnostic's SHA-256 is `a12f7c4328cc21f429056ab94ca2ee575af7b67f7db39032db83646860ce315b`.
It shows that adding state-conditioned rows did not yet produce robust sequential control;
it is not an AndroidWorld, AITW, BrowserGym, OSWorld, AppWorld, MCPMark, EnterpriseOps-Gym,
real-email/Notion, screenshot-grounding, or hardware-throughput score.

### Corrected stateful trajectory gate and weight audit (v14)

The v14 continuation adds 23 balanced state-conditioned rows, including browser open/click/type/
Enter variants, and preserves the v11 parent-head initialization.  A code audit found three
deployment/harness issues that are now fixed: lexical mobile selection was reading the goal instead
of only the next action, state JSON polluted quoted argument candidates, and the trajectory scorer
compared runtime metadata to the expected `{tool,args}` object.  The independent validator now
projects runtime responses to `{tool,args}` and keeps the state-transition check separate.

The weight audit is consistent with probe transfer: all 40 shared backbone tensors are unchanged
(relative L2 `0.0`), while the route and dense-selector probes move by relative L2 `0.2548` and
`0.2580`.  The v14 train selector is `89.66%`; held-out mobile and productivity selector scores
remain `60%` and `75%`, so the added rows have not improved the external holdouts.

The corrected in-app WebGPU run reports:

| Condition | Schema-valid | Exact `{tool,args}` | Closed-loop | Complete pass@1 |
| --- | ---: | ---: | ---: | ---: |
| Guarded dense | 12/13 | 4/13 | 3/13 | 0/3 |
| No-guard dense | 11/13 | 1/13 | 1/13 | 0/3 |
| No-guard retrieval sidecar | 8/13 | 2/13 | 1/13 | 0/3 |

The guarded condition improves because the lexical selector is now scoped to the requested action,
and the corrected grounder recovers app names such as `Gmail` and `Notion` instead of state keys
such as `app`.  The learned dense selector still confuses browser open/click and later mobile
actions; the retrieval sidecar is an ablation, not neural accuracy.  The hash-bound v14 checkpoint,
bundle, provider condition, first failures, and weight audit are in the [`m16 receipt`](paper/results/raw/m16-webgpu-mobile-productivity-trajectories-v14.json),
SHA-256 `9ad33d0d5d1613cb7c4f024792412dd61d4a6b165c3039163cd1e83902976dc8`.

This is stronger local evidence than the earlier M15 receipt, but it remains a negative workshop
gate: no official Android emulator, AgentNetBench, BrowserGym/OSWorld VM, MCP server, screenshot
grounding, real account, or hardware-throughput run has been completed.

### Stateful email/Notion/browser transfer probe (m55)

The reusable [`stateful_productivity.py`](../src/localagent/data/stateful_productivity.py) contract
now gives SFT, pointer supervision, selector probes, and future RL one canonical local state
machine.  It contains disjoint train/evaluation slots and phrasing for five workflows: complete
Gmail compose/send, Notion page creation, browser search, browser 404 recovery, and no-tool
abstention.  The evaluation split has 16 ordered decisions over a 62-tool WebGPU catalog.  Every
step is scored independently for schema validity, tool/argument exactness, state transition, and
closed-loop completion; final task state and recovery are scored separately.

The [`m55 receipt`](paper/results/raw/m55-stateful-productivity-transfer-v1.json) compares the
frozen 10.5M-parameter BPE backbone with a seed-matched random backbone.  Both arms receive the
same 160 route/selector updates and pointer-copy budget.  The pretrained arm reaches selector
top-1 `53.33%` and top-3 `80.00%`, versus `46.67%` and `86.67%` for the random arm.  Closed-loop
success is only `1/16` (`6.25%`) for both arms, with `0/5` complete workflows and `0/1` recovery;
the only successful episode is abstention (`1/1`).  This is a useful negative result: better
teacher-forced selection does not yet imply stateful execution, so the next training target is
argument/state grounding and recovery rather than more route-head fitting.

The frozen shaped-reward projection (schema `0.10`, tool `0.25`, arguments `0.25`, transition
`0.25`, terminal `0.15`) averages `0.2500` for the pretrained arm and `0.2656` for the random
control, another warning that partial local signals can move independently of task completion.

The probe is synthetic and local by design: it uses no public benchmark task text, screenshots,
emulators, MCP servers, external APIs, real email, or real Notion account.  It is a training and
diagnostic contract, not an AndroidWorld, BrowserGym, MCPMark, or publication score.

### Stateful productivity low-rate transfer ablation (m56)

The [`m56 receipt`](paper/results/raw/m56-stateful-productivity-transfer-ablation-v1.json) adds the
missing weight-adoption control.  It keeps the same five disjoint workflows, 16 evaluation
decisions, 62-tool catalog, seed, pointer vocabulary, and closed-loop evaluator while comparing:

| Arm | Selector top-1 | Selector top-3 | Closed-loop | Complete workflows | Backbone relative ΔL2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained, frozen | 53.33% | 73.33% | 1/16 | 1/5 | 0 |
| Pretrained, low-rate unfrozen (`1e-5`) | 73.33% | 80.00% | 1/16 | 1/5 | 0.188% |
| Matched random backbone | 33.33% | 80.00% | 1/16 | 1/5 | not applicable |

The low-rate arm improves held-out selector top-1 by `20.00` percentage points over the frozen
arm and `40.00` points over random, while changing the mixer/FFN/embedding groups by only
`0.41%`/`0.41%`/`0.046%` respectively.  It does **not** improve closed-loop execution, recovery
(`0/1`), or complete workflows; the only completed task remains abstention.  The appropriate
weight policy is therefore: retain the verified BPE backbone as a low-rate initialization for
future native adaptation, but do not promote it as evidence of stateful agent capability.  The
next decisive experiment is a larger public-train continuation with native BrowserGym/MobileGym
or MCPMark execution, not more selector-only tuning on this five-task mock.

This remains a synthetic local transfer ablation.  It reads no public benchmark task text,
screenshots, emulator, MCP server, real email, or Notion account and must not be reported as an
official benchmark score.

### ToolSandbox public-source projection and transfer probe (m59)

ToolSandbox is the most direct public stress test for the requested stateful productivity surface:
its scenarios cover state dependency, canonicalization, multiple tool calls, multi-turn
clarification, and insufficient information.  The [`m59 receipt`](paper/results/raw/m59-toolsandbox-public-projection-transfer-v1.json)
comes from a pinned upstream checkout and a static-AST adapter, producing 107 train and 20
held-out canonical `Conversation` rows over 32 candidate tools.  The adapter retains the public
scenario request and candidate list but reads no verifier, imports no simulator, executes no tool,
and calls no external API.

Starting from the transferred m56 child, 32 low-rate SFT updates improve held-out teacher-forced
token accuracy from `65.31%` to `71.53%` and reduce mean loss from `2.766` to `2.139`; sequence
exactness remains `0/20`.  A candidate-list dense-selector probe raises top-1 from `45%` for the
inherited selector to `75%` after retraining, but a matched random backbone also reaches `75%`.
The result supports the data adapter and selector retraining protocol, not representation transfer
or stateful execution; the adoption decision is explicitly `do_not_adopt_as_representation_evidence`.

### ToolSandbox schema-conditioned transfer probe (m61)

The next pass keeps the same task-disjoint split but enriches each candidate tool with a static
JSON schema: function signatures, primitive annotations, required arguments, and the first
docstring summary are parsed from the pinned `tool_sandbox/tools/*.py` files with Python's AST.
The adapter never imports ToolSandbox, resolves decorators, reads verifiers, executes tools, or
contacts an API.  This is the right interface for the requested email/Notion-style deployment:
the model sees the task plus the actual candidate schema instead of a name-only placeholder.

The [`m61 receipt`](paper/results/raw/m61-toolsandbox-schema-conditioned-transfer-v1.json) binds
38 statically extracted functions, 107 train rows, and 20 held-out rows.  The transferred
backbone's teacher-forced metrics are unchanged from m59 (`65.31% → 71.53%` token accuracy;
sequence exactness `0/20`) because this SFT path predicts the assistant call rather than training
the schema text itself.  The schema-aware candidate selector improves top-3 coverage from `85%`
to `100%`, while top-1 is `75%`; the matched-random backbone also reaches `75%` top-1.  Therefore
schema conditioning is adopted as a deployment interface improvement, but pretrained
representation reuse is still not promoted as a quality claim.  Native ToolSandbox/MCPMark
execution and state verifiers remain required for any real side-effect or productivity score.  On
the untouched pinned MCPMark descriptions, the same child remains at `20.71%` standard routing,
`2.37%` selector top-1, and `14.79%` top-3 (`22.86%`, `1.43%`, and `14.29%` on easy), exactly
matching m60's m59 arm.  This negative transfer is important: local schemas improve candidate
coverage, but do not solve cross-service Notion/filesystem/Postgres schema retrieval.

### ToolSandbox checkpoint-in-loop text decoder evaluation (m63)

The selector-only probe was insufficient to answer whether the deployed WebGPU dispatch path can
produce a usable call.  The [`m63 receipt`](paper/results/raw/m63-toolsandbox-text-decoder-eval-v1.json)
now runs the checkpoint-backed constrained decoder on all 20 held-out rows.  Each row supplies its
retained public candidate list and static JSON schemas; the evaluator runs route, character
retrieval, pointer grounding, parsing, and recursive schema validation, then compares the call to
the AST-extracted target.

With the schema-child checkpoint, restricting retrieval to the row's candidate list gives `100%`
route coverage, `60%` tool exactness, `30%` exact tool-and-argument matches, and `100%` schema
validity.  Canonicalization is `64.29%` tool exact / `35.71%` argument exact; state-dependency rows
fall to `25%` / `25%`.  Exposing the checkpoint's inherited global selector instead of the row
candidate contract collapses to `10%` tool exactness and `10%` schema validity.  The matched
no-schema child ties the row-retrieval arm exactly, so schema extraction is not yet evidence that
the pretrained representation was adopted; it is an interface and safety contract only.

This is a stronger decoder-in-the-loop diagnostic than m59/m61 teacher forcing, but it remains a
static text projection: no ToolSandbox simulator, user simulator, milestone verifier, MCP service,
official split, native browser, or external side effect ran.  The practical deployment decision is
to require task-scoped candidate retrieval and schema validation before dispatch, while treating
stateful execution and exact argument grounding as unresolved training targets.

### Local WebGPU and Hugging Face export receipts (m57–m58)

The same m56 child was exported to a clean static-demo bundle and an independent Hugging Face
format bundle.  The [`m57 receipt`](paper/results/raw/m57-stateful-webgpu-deploy-verification-v1.json)
binds the 10,524,544-parameter checkpoint, tokenizer, eight generated inference artifacts, and
their exporter manifest.  The fp32 and fp16 ONNX graphs pass the hard CPU parity gate (maximum
fp32 logit difference `6.68e-6`; fp16 logit difference `7.36e-3`; fp16 argmax agreement `1.0`),
and the clean static app contains all required files.  The [`m58 receipt`](paper/results/raw/m58-stateful-hf-local-export-v1.json)
binds the local `model.safetensors`, serialized heads, tokenizer, config, and README hashes.

These are local artifact and deployment checks, not native WebGPU capability measurements.  The
Hub is not authenticated in this environment, so neither receipt claims a public model/Space URL
or upload; a user-provided write token and repository namespace are still required.  No external
email, Notion, browser, MCP, or desktop side effect was executed.

### EnterpriseOps-Gym public email retrieval diagnostic (v10)

To probe realistic enterprise tool breadth without contaminating training or claiming an execution
score, the frozen v10 dense selector was evaluated on the 67 email task IDs exposed in both the
public `oracle` and `plus_15_tools` configs at HF revision
`c8e538eae8a6205294f0a86675fefdc1fac408f6`.  The adapter consumes only task/domain/system/user
text and selected tool names.  It drops `verifiers` and `gym_servers_config`, deduplicates repeated
candidate names in memory, generates name-only descriptions, and never executes MCP servers or
state transitions.  The two raw JSON payloads remain outside Git; their sizes and SHA-256 hashes
are bound in the [`m14 receipt`](paper/results/raw/m14-enterpriseopsgym-email-retrieval-v10.json),
whose tracked SHA-256 is `b4fc14edc5e87f6bec14db52b21f90eee0ab8beb34ab4e6f215444c68429d669`.

| Metric | Result |
| --- | ---: |
| Records | 67 email tasks |
| Oracle tools (mean) | 5.69 |
| Candidate tools (mean) | 20.00 |
| Retrieval hit@1 | 26.87% (18/67) |
| Retrieval hit@3 | 56.72% (38/67) |
| Retrieval hit@5 | 79.10% (53/67) |

The dominant false positives were generic `update_label` (21), `get_user_profile` (12), and
`send_message` (5), showing that the current dense selector is biased toward frequent tool names
when descriptions contain no schemas or examples.  This is useful failure evidence for the next
training pass—schema-conditioned descriptions, calibrated abstention, and stateful local MCP
execution—but it is explicitly not an EnterpriseOps task-success, verifier, or leaderboard score.

### v10 no-guard dense WebGPU continuation

The next continuation starts from the v9r retrieval-sidecar child and keeps the same 10.5M
parameter BPE model, tokenizer, and 62-tool catalog.  Its 3,000-step report covers 114 normalized
rows (86 repeated synthetic mobile rows plus bounded AndroidControl/AITW rows) and holds out two
mobile episodes and four productivity rows.  Route accuracy is `100%`; the held-out mobile dense
selector is `6/10` (`60%`) and the held-out productivity selector is `3/4` (`75%`).  A weight audit
found 51 shared tensors, no model or tokenizer mismatch, zero backbone movement, and an action-head
relative delta L2 of `0.5294`.  This establishes compatible pretrained-weight reuse and head
movement, not a no-transfer improvement claim.

The fp32/fp16 ONNX graphs and serialized dispatch heads pass the export parity gate (fp32 logits
drift `8.58e-06`, fp32 hidden drift `6.14e-06`, fp16 argmax agreement `1.0`).  The explicit in-app
browser WebGPU run with `mobile_guard=0&selector=dense` reports `4/9` exact tools, `4/9` exact
arguments/actions, `6/9` schema-valid actions, and `4/9` closed-loop successes: mobile `2/7`,
productivity `2/2`.  Timing was not collected.  The full identity and summary hash are in the
[`m12-v10 receipt`](paper/results/raw/m12-webgpu-mobile-productivity-v10.json).

This result is a clearer no-guard reproduction than the retrieval ablation, not a higher-accuracy
result: the learned dense selector remains the limiting component.  It is still synthetic,
text-first, single-concurrency state evaluation with no screenshot grounding, trusted OS input,
real accounts, official environment runner, or hardware throughput measurement.  The retrieval
sidecar's earlier `9/9` result must remain reported separately from this learned policy.

### Pretrained-head transfer control

The dispatch runner now exposes an explicit `--probe-init` switch so transfer is tested rather
than assumed.  A matched 3,000-step replay from the v9r parent compared the inherited route and
dense-selector heads with seeded-random probe initialization; the backbone, tokenizer, data bytes,
holdout IDs, optimizer settings, and seed were held constant.  Parent-head reuse raised training
selector accuracy from `75.86%` to `86.21%` and kept action-head movement smaller (`0.4426` vs
`1.0753` relative delta L2), but held-out selector accuracy was identical: mobile `6/10` and
productivity `3/4` for both arms.  The result supports lineage and optimization stability, not a
quality gain.  The complete hash-bound control is [`m13 transfer ablation`](paper/results/raw/m13-mobile-dispatch-transfer-ablation.json).

For comparison, an earlier AndroidControl-only baseline continued the WebGPU-tier parent
(`webgpu-10m-hybrid`, 10,524,544 parameters) for eight CPU
SFT updates at `1e-5` and `max_seq_len=2048`.  On the same normalized rows, mean assistant loss
  changed from `8.6985` to `7.9501`; assistant-token accuracy was `2.33%` and exact trajectory
  accuracy was `0%`.  These are language-model bridge metrics, not emulator task success.  The
  baseline child checkpoint SHA-256 is
  `9ea68807ac758c564200a5649f9196ecde3bccb8a66edae04eec8c6c14813eae`; its ONNX bundle manifest
  SHA-256 is `13e5549d69ab9286ab41e2c75901ee67412327487997f4740dd6c31f7f726940`.  The bundle passed
  the hard PyTorch parity gate for fp32 and fp16 logits/hidden graphs (fp32 maximum absolute drift
  `7.39e-06`, fp16 `6.08e-03`) and contains a 10.5M-parameter action graph suitable for the
  existing static WebGPU demo.  The child checkpoint and bundle are deliberately kept outside Git
  until a full held-out/runtime evaluation receipt exists.

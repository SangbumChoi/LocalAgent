# Realistic agent evaluation and data policy

This project targets a text-first WebGPU agent with a small (<100M) decoder.  Real deployments
still need to be tested against the environments people use: mobile UI control, browser actions,
desktop workflows, and stateful API/MCP tools.  The source-linked inventory is
[`configs/data/realistic-agent-eval.catalog.yaml`](../configs/data/realistic-agent-eval.catalog.yaml).
It is deliberately a catalog, not a downloader: every acquired byte must be recorded in a local
provenance manifest with an upstream revision, byte count, and SHA-256.

The current catalog contains 36 source-linked rows (four train-eligible and 32 evaluation or
restricted) and has a canonical SHA-256 fingerprint recorded by the preflight command below.
The current fingerprint is `aa7e7b90d9b8158d7bd1c76430a6ffb6bbb0ad3ceef1aaac14046684cfc93f30`.

Run the read-only readiness report before acquiring or evaluating anything:

```bash
PYTHONPATH=src python scripts/realistic_agent_preflight.py
```

The report currently identifies the four local text-first adapters as runnable and all 31
environment/evaluation rows as blocked by their pending integration status (for example, no
`adb`, Docker, VM, or upstream BrowserGym checkout).  `--strict` intentionally exits non-zero
until those external runners are installed and pinned; this is a readiness gate, not a benchmark
score.

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

The adapted child was then exported through the browser bundle path.  The fp32/fp16 logits and
hidden graphs plus the distinct hidden-only action graph passed the hard parity gate; fp32 logits
drift was `6.91e-06`, fp32 hidden drift `5.66e-06`, and fp16 logits argmax agreement was `1.0`.
In the in-app browser, an explicit WebGPU session loaded the verified 10.5M bundle and the local
9-step mobile/email/Notion state suite produced schema validity `6/9` and closed-loop success
`4/9` (mobile `2/7`, productivity `2/2`).  The WASM control had the same quality result.  The
versioned receipt is [`m11-webgpu-mind2web-bpe-child.json`](paper/results/raw/m11-webgpu-mind2web-bpe-child.json);
no hardware throughput claim is made because timing was not collected in this run.

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
- [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym): 1,115 public
  enterprise tasks across eight domains, including 104 email tasks and 512 tools.  The official card
  describes containerized execution with SQL state verifiers and the dataset API reports Apache-2.0;
  tasks/verifiers stay evaluation-only to prevent benchmark memorization.  The pinned HF revision is
  `c8e538eae8a6205294f0a86675fefdc1fac408f6`.  The card-level inventory (104 email tasks) is not
  the same as a downloadable config row count: the bounded oracle and `plus_15_tools` email
  configs used below expose 67 matching task IDs, so the receipt reports 67 and retains the
  104 figure only as inventory context.
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

The pinned [MCPMark](https://github.com/eval-sys/mcpmark) metadata profile is recorded in
[`m21-mcpmark-metadata-profile-v1.json`](paper/results/raw/m21-mcpmark-metadata-profile-v1.json).
The checkout contains 239 metadata rows: 169 `standard` L3 tasks and 70 `easy` L1 smoke tasks,
covering 38 Notion, 35 browser/Playwright, 40 filesystem, 33 GitHub, and 93 database tasks.
State types are text (133), URL (71), and video (33); the profile retains neither descriptions nor
state assets. MCPMark's own runner supports repeated runs and pass@k/pass^k aggregation, so it is
the next appropriate stability gate once the local MCP/browser dependencies are installed.

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

# Realistic agent evaluation and data policy

This project targets a text-first WebGPU agent with a small (<100M) decoder.  Real deployments
still need to be tested against the environments people use: mobile UI control, browser actions,
desktop workflows, and stateful API/MCP tools.  The source-linked inventory is
[`configs/data/realistic-agent-eval.catalog.yaml`](../configs/data/realistic-agent-eval.catalog.yaml).
It is deliberately a catalog, not a downloader: every acquired byte must be recorded in a local
provenance manifest with an upstream revision, byte count, and SHA-256.

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
- [EnterpriseOpsGym](https://huggingface.co/datasets/EnterpriseAgents/EnterpriseOpsGym): 649 public
  enterprise tasks, including 67 email tasks.  The dataset card says MIT, but the benchmark tasks and
  SQL verifiers remain held out to prevent benchmark memorization.
- [MCP-Persona](https://github.com/wwh0411/MCP-Persona): 173 personalized tool-calling tasks that
  include Notion and email MCP servers.  Its repository terms must be checked before redistribution;
  keep the simulated account state and checkpoint labels evaluation-only.
- [TUA-Bench](https://github.com/facebookresearch/TUA-Bench): 120 execution-based terminal tasks,
  including document editing, email management, and live-web information seeking.  It is CC-BY-NC
  and explicitly a benchmark, so it is not a training corpus.

The existing BFCL, WebLINX, and protected Mind2Web/browser captures in `data/private/` remain
evaluation-only under the same rule.  A public license does not make a benchmark safe to train on:
task prompts, verifier code, and gold state can directly inflate the score.

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

An 8-step continuation over the 26-row AndroidControl+AITW mixture reduced mean normalized-row
loss from `8.6239` to `7.9730` (assistant-token accuracy `2.02%`, exact trajectory accuracy `0%`).
The resulting 10.5M WebGPU bundle passed the same hard ONNX/PyTorch parity gate (fp32 max drift
`7.15e-06`, fp16 max drift `4.83e-03`).  This expands training coverage, but remains a bridge
measurement rather than evidence of environment success.

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

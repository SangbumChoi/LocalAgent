# Realistic agent evaluation research memo

Status: protocol refresh on 2026-08-03. This memo records the public benchmark methods that
matter for a sub-100M text-first WebGPU agent. It is a source and protocol guide, not a claim that
the repository has completed every benchmark.

## What the public benchmarks actually measure

| Surface | Public source and method | What LocalAgent may claim locally |
|---|---|---|
| Mobile UI | [AndroidWorld](https://github.com/google-research/android_world) runs resettable Android emulator tasks with accessibility/screenshot observations and durable task rewards. | Accessibility-tree/text protocol tests are useful for routing; an Android emulator, ADB, official task set, and reward logs are required for a native score. |
| Personalized mobile UI | [iOSWorld](https://iosworld.io/) provides 133 tasks over 26 interconnected iOS apps with persistent seeded user identity and an optional MCP server. | Treat identity, cross-app state, and MCP-vs-GUI as separate axes; a WebGPU text projection cannot claim native iOS control or personalization. |
| Cross-app mobile assistant | [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) defines 201 long-horizon tasks over 20 Android apps, including agent-user interaction and MCP-augmented workflows, with screenshot/accessibility/backend-state observations and deterministic verification. | Freeze the source revision, rooted AVD/Docker image, credentials, and task manifest. macOS lacks the upstream Linux/KVM runtime; source parsing is provenance only, not native mobile/email/browser success. |
| Mobile safety | [MobileSafetyBench](https://mobilesafetybench.github.io/) evaluates Android-device safety, harmful side effects, and indirect prompt injection in messaging/banking-style tasks. | Run a dedicated refusal/confirmation/safe-side-effect gate before enabling email, messaging, settings, or payment tools. |
| Visual prompt-injection safety | [VPI-Bench](https://github.com/cua-framework/agents) reports 306 dynamic cases across Amazon, Booking, BBC, Messenger, and Email, separating attempted from successful malicious actions. | Keep attack pages and traces evaluation-only; a text-only WebGPU model can measure refusal/confirmation policy but cannot claim visual robustness without a vision-capable runner. |
| Contextual-integrity safety | [AgentCIBench](https://github.com/UKPLab/arxiv2026-agentcibench) uses state/prompt/recipient/`must_share`/`must_not_share` scenarios and engagement-conditioned leakage. | Add disclosure restraint and consent metrics alongside task success; generated scenario pools and judge configuration must be release-pinned. |
| Curated mobile control | [AndroidControl-Curated](https://github.com/batechworks/AndroidControl_Curated) purifies AndroidControl task ambiguity and reports a matched curated split. | Compare original and curated tasks with identical identities/seeds; do not mix the curated benchmark's evaluation rows into SFT. |
| Dynamic mobile device control | [AndroidLab](https://github.com/MadeAgents/mobile-use/tree/main/benchmark/android_lab) is an emulator/ADB benchmark surfaced by MobileUse, with patched Clock/Settings tasks and a release-dependent AndroidLab submodule. | Freeze the submodule revision, task manifest, Docker/AVD image, and license before using it as a score; a text-only WebGPU projection cannot claim screenshot or emulator success. |
| Personalized/proactive mobile | [KnowU-Bench](https://github.com/ZJU-REAL/KnowU-Bench) reports 192 registered tasks over 23 apps, with hidden profiles, exposed behavioral logs, and an online user simulator for clarification and consent. | Treat preference acquisition, intervention calibration, and consent as separate metrics; Docker/KVM and API-backed user simulation are required for a native result. |
| Smartphone agent benchmark | [AppAgent](https://github.com/TencentQQGYLab/AppAgent) releases the smartphone evaluation benchmark used by its tap/swipe agent and supports Android emulators or devices. | Pin the benchmark release, APK hashes, device image, and exact task manifest; text-only action vocabulary is not smartphone task success. |
| Browser | [BrowserGym](https://github.com/ServiceNow/BrowserGym) exposes MiniWoB, WebArena, WorkArena, VisualWebArena, WebLINX, and related environments through Gymnasium/Playwright. | DOM/accessibility grounding can be tested in the existing MiniWoB runner; screenshot or live multi-site claims require the matching runtime and task release. |
| Real web trajectories | [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) contains more than 2,000 tasks across 137 websites and 31 domains, with public training data and protected test splits. | Train only on the public train partition; keep test tasks and canary strings out of all corpora. Report DOM/action replay separately from native live-web success. |
| Realistic live-site browser workflows | [WebBench](https://github.com/Halluminate/WebBench) covers 2,454 READ/CREATE/UPDATE/DELETE/file tasks across 452 live websites, including authentication, forms, and downloads. | Keep the live task set and credentials evaluation-only; use DOM/action safety canaries or a vendor-approved resettable environment rather than training on the tasks. |
| Contamination-resistant browser canary | [BU Bench V1](https://github.com/browser-use/benchmark) contains 100 encrypted tasks drawn from WebBench, Mind2Web 2, BrowseComp, GAIA, and custom challenges. | Preserve encryption and task text out of SFT and published artifacts; only run a release-matched browser evaluation with decrypted data held locally. |
| Desktop computer use | [OSWorld](https://github.com/xlang-ai/OSWorld) uses real desktop VMs and execution-based evaluators; [OSWorld 2.0](https://osworld-v2.xlang.ai/) adds 108 long-horizon workflows with dynamic environments, cross-source reasoning, implicit state, and visual precision. | A text/accessibility projection is a diagnostic only. A publication score needs the release-matched VM, assets, initial state, action log, and evaluator. |
| Screenshot/action trajectories | [OpenCUA AgentNet](https://github.com/xlang-ai/OpenCUA) provides cross-OS computer-use trajectories; AgentNetBench is an offline representative task suite. | The m47–m62 projection measures action priors and text routing only. Visual grounding requires the embedded images and the upstream AgentNetBench evaluator. |
| Desktop grounding | [GroundCUA](https://github.com/ServiceNow/GroundCUA) publishes 56K screenshots and 3.56M+ human-verified element annotations across 87 desktop applications. | Keep screenshot/box training and evaluation separate from this text-first model; any use requires dataset-term review and a visual encoder. |
| GUI action protocol | [UI-TARS](https://github.com/bytedance/UI-TARS) documents separate computer-use, mobile-use, and grounding contracts, including long-press, app launch, home, and back actions. | Reuse only the schema/action vocabulary for compatibility probes; coordinate or screenshot claims require a vision-capable model and release-matched runtime. |
| GUI plus MCP control | [OSWorld-MCP](https://github.com/X-PLUG/OSWorld-MCP) measures GUI actions, MCP invocation, and decisions together across 158 validated tools and seven desktop applications. | Report tool-invocation rate separately from GUI completion; a schema-routing probe is not an OSWorld-MCP score. |
| Stateful local tools | [Apple ToolSandbox](https://github.com/apple/ToolSandbox) evaluates stateful, conversational tool execution with a user simulator and milestone DAGs, including state dependency, canonicalization, and insufficient-information cases. | Static AST rows can train schemas and measure constrained dispatch; only the simulator plus milestone verifier can establish ToolSandbox success. |
| Stateful MCP services | [MCPMark](https://github.com/eval-sys/mcpmark) runs isolated Notion, GitHub, filesystem, Postgres, and Playwright services with strict verification. Its current repository identifies MCPMark Verified as the default task set. | Tool/schema retrieval and local state contracts are preflight evidence. Native claims require the matching verified release, isolated services, verifier output, and pass@k aggregation. |
| Enterprise email/tools | [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) exposes large tool catalogs and stateful SQL-verifier workflows across enterprise domains. | Name/schema retrieval is a useful failure profile; task success requires the containerized MCP servers and SQL verifiers. |
| Verifiable computer-use training | [CUA-Gym](https://huggingface.co/datasets/xlangai/CUA-Gym) publishes a CC-BY-4.0 task table with 10,910 desktop/web/cross-app instructions and executable setup/reward artifacts. | Its metadata table has one `train` split and no official held-out evaluation split; use it for coverage and sandboxed RLVR only after task-identity holdout, artifact review, and a disposable runtime. The current receipt intentionally consumes metadata only. |
| Public desktop trajectory archives | [OSWorld 2.0 trajectories](https://huggingface.co/datasets/xlangai/osworld2.0-trajectory) and [OSWorld-Verified trajectories](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs) provide large model-run packages. | They are evaluation/provenance sources until task identity and split leakage are resolved; archive screenshots and verifier outputs are not silently admitted to WebGPU SFT, and native OSWorld evidence still needs the release-matched VM. |

### Official-source audit and admission boundary (m208)

The [`m208 receipt`](paper/results/raw/m208-realistic-evaluation-source-audit-v1.json) freezes the
historical research inventory against the 40-row canonical catalog and 24-row supplemental registry.
It cross-checks the official contracts for AndroidWorld, AndroidControl/AITW, MobileSafetyBench,
iOSWorld, AppAgent, BrowserGym, Mind2Web, WebBench/BU Bench, OSWorld, AgentNet, GroundCUA/UI-TARS,
ToolSandbox, MCPMark, EnterpriseOps-Gym, and MobileGym.  The audit makes the deployment boundary
explicit: only AndroidControl, AITW, xLAM function calling, and the public Mind2Web train partition
are currently train-eligible; benchmark task text, screenshots, emulator/VM assets, credentials,
MCP state, and verifier outputs remain evaluation-only.

This matters for the requested email, Notion, browser, and computer-use scenarios.  MCPMark and
EnterpriseOps-Gym require isolated service state and independent verification; iOSWorld and
MobileSafetyBench require seeded mobile runtimes plus safety/personalization metrics; BrowserGym and
OSWorld require release-matched browser/VM execution.  A compact WebGPU text projection may measure
route, retrieval, grounding, schema validity, abstention, and action history, but the receipt does
not convert those proxies into native or leaderboard scores.

### Matched multi-surface continuation and native bridge (m209–m211)

The [`m211 receipt`](paper/results/raw/m211-multisurface-continuation-native-bridge-v1.json) trains
one matched 64-step continuation over 4,752 public rows from AndroidControl, AgentNet, Mind2Web,
and the ToolSandbox AST projection, with 1,069 source-disjoint held-out rows.  The warm child rises
from `64.31%` to `72.30%` aggregate assistant-token accuracy, while the random-backbone control
rises from effectively `0%` to `46.55%`.  Warm movement remains small (embedding `0.97%`,
mixer `0.29%`, FFN `0.38%`, normalization `0.02%`) and the action heads move `0%`; exact sequence
accuracy remains `0%` for both arms.  The gains are therefore representation-transfer evidence,
not complete tool policy learning.

The native bridge is unchanged: both children complete `1/5` bounded interactive ToolSandbox
scenarios.  Random improves one ambiguous-contact similarity from `0.0` to `0.3333`, but neither
arm gains a new fully verified task.  The warm child is retained as a compatibility diagnostic,
not exported or adopted.  Reliable WebGPU email/Notion/browser control still requires jointly
training the runtime-aligned route, candidate selector, argument grounding, and state/action-history
policy, followed by native closed-loop verification.

The m156 follow-up probe uses only CUA-Gym's public instruction field and its metadata `platform`
label (`desktop`, `web`, or `cross_app`) on a deterministic task-ID holdout.  A frozen m142 warm
checkpoint reaches 81.37% surface accuracy versus 77.87% for the matched random control, with
exactly zero backbone movement because only a new linear probe is trained.  This is evidence that
the warm representation carries a small amount of broad deployment-surface signal; it is not an
action label, task-success, screenshot-grounding, or native browser/desktop result.  See the
[`m156 receipt`](paper/results/raw/m156-cua-gym-surface-probe-v1.json) and reproducible
[`train_cua_gym_surface_probe.py`](../scripts/train_cua_gym_surface_probe.py).

The m157 browser continuation uses the public [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web)
train split, enriches each action with a bounded DOM-candidate snapshot, and holds out three whole
parent records plus typed slot values.  After 64 matched updates, the warm pointer head reaches
15/63 exact spans versus 0/63 for the matched random body; both begin at 0/63.  The warm body moves
0.90% in embeddings, 0.50% in attention/mixer, and 0.55% in FFN, while the pointer/action heads
move 46.40%.  The pointer vocabulary grows from 17 to 19 rows by design.  This is a small
in-source action-replay diagnostic—not the official Mind2Web test score or native BrowserGym,
email, Notion, MCP, or browser-account control.  See the [`m157 receipt`](paper/results/raw/m157-mind2web-grounded-transfer-v1.json).

The m158 mobile continuation returns to the current m142 parent and uses a public AndroidControl/AITW
train projection with a disjoint 904-row evaluation file.  Focused oversampling produces an offline
selector score of `41.26%` top-1 with `100%` routing and `25%` exact pointer spans, while the inherited
body is unchanged and only action heads move substantially (`124.77%` relative ΔL2).  A pinned
MobileGym first-20-task canary reaches `1/20` (`5%`), exactly matching the parent same-range result.
This cleanly separates offline action-format learning from state-conditioned simulator transfer;
it is not the full MobileGym score, Android emulator/screenshot grounding, or a browser/email/Notion
side-effect claim.  See the [`m158 receipt`](paper/results/raw/m158-mobile-dispatch-transfer-v1.json).

The m159 MCPMark continuation repeats the public trajectory-log comparison from the current m142
parent.  Eight redacted filesystem, Notion, GitHub, Postgres, and Playwright rows train and two
parent-disjoint Playwright rows remain held out.  Warm initialization improves held-out assistant
token accuracy from `39.38%` to `43.18%`; a matched random backbone reaches only `3.12%` after the
same 32 updates, a `+40.06` point gap.  Both arms remain at `0%` exact multi-turn sequence
accuracy.  Warm movement stays below `0.5%` in each inherited backbone group, supporting the
low-rate body/high-rate adapter recipe while rejecting any claim of native MCP, email, Notion, or
browser side effects.  See the [`m159 receipt`](paper/results/raw/m159-mcpmark-current-parent-transfer-v1.json).

The recurring methodological point is that a benchmark is not just a prompt list. The authoritative
score includes the observation contract, reset state, action interface, environment revision, and
verifier. A static conversation projection must be labelled as such even when it uses the original
task text.

### MobileGym and OSWorld-V2 source audit (m121)

The [`m121 source-audit receipt`](paper/results/raw/m121-mobilegym-osworld-source-audit-v1.json)
hash-pins two public benchmark contracts without converting either into training data.  The pinned
[MobileGym](https://github.com/Purewhiter/mobilegym) revision describes 28 simulated mobile apps,
416 parameterized templates, and a 256-task held-out test split (160 train templates by subtraction),
with structured state snapshots, tap/type/swipe-style actions, deterministic state-diff judges, and
an AnswerSheet protocol.  Its code is Apache-2.0, but the companion benchmark data is CC-BY-NC-4.0;
the receipt therefore records the release pointer while leaving the approximately 1.9 GB archive
out of the repository and out of SFT.

The same receipt binds the active [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2) release
`osworld-v2-2026.06.24`, its 108-task hash manifest, gated task and asset datasets, website/code
tags, and provider image digest.  The gated task implementations, desktop VM, action logs, and
verifier were not acquired here, so this is provenance and protocol evidence—not an OSWorld-V2
score, desktop-control result, or justification for publishing benchmark-derived task text.

### Mobile/GUI source audit (m163)

The [`m163 source-audit receipt`](paper/results/raw/m163-mobile-grounding-source-audit-v1.json)
binds current source revisions and aggregate contracts for KnowU-Bench, AppAgent, GroundCUA, and
UI-TARS.  The audit keeps their task text, hidden profiles, screenshots, annotations, APKs, and
model traces out of the repository.  KnowU-Bench is the most important addition for realistic
personal assistance: preference inference, clarification, proactive intervention, consent, and
silence are evaluated separately from ordinary GUI execution.  AppAgent supplies a small
45-task/9-app smartphone gate, while GroundCUA is a vision-only grounding resource and UI-TARS
supplies a useful mobile/desktop action vocabulary.

This is deliberately stronger provenance but not a capability result: the receipt has no native
emulator/desktop runs, no WebGPU run, no training rows, and no official scores.  For the current
text-first model, use only compact state/action contract projections; screenshot grounding,
hidden-profile reasoning, and APK interaction require their release-matched runtimes and
additional model inputs.

### MobileWorld source and runtime preflight (m302)

The [`m302 receipt`](paper/results/raw/m302-mobileworld-source-runtime-audit-v1.json) pins the
public [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) checkout at revision
`0dcd0980eac64d76f498f93568a1ec0594b743c4`.  AST inventory finds all `201` upstream task classes
across `20` apps: calendar, Chrome, Gmail, mall, maps, Mastodon, messages, native apps, settings,
and work.  The receipt records the source-file hashes and the contract's screenshot,
accessibility-tree, backend-state, tap/swipe/type/keyevent/wait/MCP interfaces without copying task
rows into LocalAgent training.

MobileWorld is especially relevant to the requested email, browser, cross-app, and agent-user
workflows, but its official runner requires a privileged Docker-in-Docker rooted Android AVD,
Linux/WSL2 KVM, model/user-agent credentials, and optional MCP credentials.  The current host is
macOS arm64 with no Docker, ADB, QEMU, or `/dev/kvm`; the optional `uv run --offline mw env check`
also stops before preflight because `gradio==5.49.1` is not cached.  No official runner, native
environment, model, MCP service, or user simulator was executed, so the receipt deliberately has
`score: null` and remains provenance/runtime evidence rather than a MobileWorld benchmark result.

### MobileGym official split profile (m122)

The [`m122 split-profile receipt`](paper/results/raw/m122-mobilegym-source-split-profile-v1.json)
now verifies the exact split manifests from the hash-pinned MobileGym source archive without
retaining task prompts.  `train.txt` contains 160 IDs and `test.txt` contains 256 IDs with no
overlap, yielding all 416 unique benchmark tasks.  The separate seven-task payment and fourteen-
task high-risk lists are safety subsets, not additional train/test rows; their overlap is recorded
explicitly.  Family counts and split-file hashes are reproducible through
[`profile_mobilegym_source.py`](../scripts/profile_mobilegym_source.py).  This establishes a clean
evaluation boundary for the WebGPU model, but the CC-BY-NC-4.0 content, simulator state, judges,
and screenshots remain outside training and no native score is claimed.

### MobileGym native runtime smoke (m123)

The [`m123 runtime receipt`](paper/results/raw/m123-mobilegym-native-runtime-smoke-v1.json)
boots the pinned MobileGym source in a local Vite server and loads it in headless Chromium.  The
page returns HTTP 200, exposes the documented `window.__SIM__` bridge with `{os, apps}` state,
and the upstream registry loads 423 task classes while every official 160-train/256-test ID
resolves.  Repeated resets preserve state shape and size but are not byte-identical: the receipt
records timestamp-only diff paths instead of calling the reset deterministic at the raw-byte level.
The smoke invokes no model, task judge, screenshot scorer, or external account, so it is a native
runtime preflight rather than a MobileGym-Bench result.

### MobileGym text-only model probe (m124)

The [`m124 probe receipt`](paper/results/raw/m124-mobilegym-model-probe-v1.json) runs the current
10.52M deployment-repair checkpoint inside the same pinned simulator on one ID from the official
256-task test split.  The bridge supplies only a bounded DOM-text projection (no screenshot) and
translates the additive `mobile_*` contract into native MobileGym actions.  The model was invoked
twice and produced two `mobile_input_text` calls, but the task judge passed `0/1`; output and
argument values are hash-bound and omitted.  This is useful failure evidence for the current
text-first boundary, not a MobileGym benchmark receipt: `native_receipt_eligible` is explicitly
false and the strict gate remains blocked until the full release-compliant runner is supplied.

The new mobile and MCP suites reinforce the same design requirement. iOSWorld's persistent identity
and cross-app data make memory/state tracking first-class rather than an optional prompt feature;
MobileSafetyBench makes confirmation, refusal, and prompt-injection handling measurable; and
OSWorld-MCP separates the decision to invoke a structured tool from the GUI action itself. These
should be represented as distinct WebGPU heads/metrics, not collapsed into token accuracy.

The m178 source audit pins iOSWorld at `e91f4cb2ef4c9dd48fef83a894477b41fd5e209d` and
MobileSafetyBench at `bc5e0579626a280c4f551261abcb721442ff92ea`, retaining README and available
license hashes plus aggregate contracts.  This makes the safety and personalization references
reproducible without importing protected task text, seeded profiles, APKs, or simulator state into
training.  See the [`m178 receipt`](paper/results/raw/m178-mobile-safety-personalization-source-audit-v1.json).

The follow-up [`m179 manifest audit`](paper/results/raw/m179-mobile-evaluation-manifest-audit-v1.json)
temporarily fetched only the public task manifests to bind their bytes and aggregate counts:
iOSWorld has 133 task rows, while MobileSafetyBench exposes a 90-row source table plus a separate
three-row QA file.  The paper's 100-task MobileSafetyBench suite remains the benchmark contract;
the files are not interchangeable with that score and their prompts are not admitted to training.

## Recommended WebGPU protocol

The deployment should be evaluated in two non-interchangeable tracks:

1. **Offline contract track.** Use public train data only for SFT. For held-out rows, expose the
   task plus compact DOM/accessibility/state text and the task-scoped candidate tool list. Run the
   route → retrieval → pointer grounding → parser → recursive JSON-schema validator. Report route,
   tool exactness, exact arguments, schema validity, abstention, and complete-trajectory success.
2. **Native runtime track.** Run the unchanged checkpoint in the official Android, BrowserGym,
   desktop VM, ToolSandbox, MCPMark, or EnterpriseOps-Gym environment. Freeze the source revision,
   task IDs, reset seeds, browser/VM/container versions, model/tokenizer hashes, full action logs,
   and verifier output. Do not convert the offline track into a native score.

For the current WebGPU model, the first track is now implemented for ToolSandbox in
[`evaluate_toolsandbox_text.py`](../scripts/evaluate_toolsandbox_text.py). The m63 receipt shows
why the candidate contract is necessary: row-scoped retrieval achieves `60%` tool exactness,
`30%` exact arguments, and `100%` schema validity, while exposing the inherited global selector
achieves only `10%`, `10%`, and `10%`. The matched no-schema child ties the row-scoped arm, so the
schema-conditioned SFT child is not promoted as representation-transfer evidence.

The current checkpoint's public ToolACE action-history canary is [`m310`](paper/results/raw/m310-current-toolace-action-history-canary-v1.json).
It uses eight source rows from the pinned ToolACE projection and the same catalog-constrained
dispatcher used by the WebGPU path: 17 action steps produce `11.76%` tool exactness, `0%`
argument/step/episode exactness, and `70.59%` schema validity. The canary makes the failure mode
explicit—arbitrary public tool names are frequently mapped to a nearby schema or dropped—while
keeping all tool calls local and side-effect free. It is a bounded public-projection diagnostic,
not official ToolACE/BFCL execution or evidence of email, Notion, browser, or MCP control.

The larger [`m309` probe](paper/results/raw/m309-current-toolace-action-history-v1.json) covers 64
held-out rows and 113 action steps: tool exactness is `22.12%`, argument/step exactness `2.65%`,
schema validity `69.91%`, and complete-episode exactness `0%`. This is the current checkpoint's
source-projected baseline; the eight-row m310 canary is retained as the exact before/after slice
for the selector transfer below.

The follow-up [`m312 selector transfer`](paper/results/raw/m312-current-toolace-selector-free-run-transfer-v1.json)
uses 16 source-disjoint public train actions to update only the dense selector, with the decoder
backbone and route/pointer heads frozen. On the same eight-row held-out canary, selector top-1 and
free-run tool exactness both move from `11.76%` to `17.65%` (`+5.88` points), while schema
validity drops from `70.59%` to `64.71%`; argument, step, and episode exactness remain `0%`.
The matched random selector control and selector movement are retained in the receipt. This is
evidence for a candidate selector adapter—not a deployable policy—so the full-policy promotion
decision remains rejected pending larger source-disjoint and native verifier-backed runs.

The larger [`m313 transfer`](paper/results/raw/m313-current-toolace-selector-64-transfer-v1.json)
uses 447 public train actions and 113 held-out action steps. Selector top-1 rises from `23.89%`
to `30.97%` (matched random `28.32%`); free-run tool exactness rises `22.12%`→`26.55%`,
argument/step exactness `2.65%`→`7.08%`, episode exactness `0%`→`3.12%`, and schema validity
`69.91%`→`72.57%`. This is the strongest current public ToolACE transfer signal, but the
selector moves substantially (query `58.53%`, tool `225.42%`) and the evaluation remains a
64-row projection; retain it as a candidate adapter and require a larger held-out and native
verifier-backed replication before deployment adoption.

The [`m316 pointer transfer`](paper/results/raw/m316-current-toolace-pointer-free-run-transfer-v1.json)
isolates the remaining argument-grounding bottleneck. It trains 182 locatable spans from the same
public ToolACE train projection and evaluates 63 held-out spans. Offline decoded-value exactness
rises from `0%` to `19.35%`, but it only ties the matched random pointer; in the 113-step free run,
argument/step/episode exactness remains `2.65%`/`2.65%`/`0%`, schema validity rises `69.91%`→`76.99%`,
and tool exactness falls `22.12%`→`21.24%`. The pointer is therefore rejected for deployment
promotion. The adapter also now preserves unnamed legacy pointer rows under deterministic
placeholders when a checkpoint omits `ptr_args`, rather than silently misaligning or dropping
the learned matrix.

## Weight-adoption decision rule

Adopt a pretrained backbone only when all three arms use the same tokenizer, configuration, seed,
data rows, and training budget:

- pretrained frozen or head-only continuation;
- pretrained low-rate-unfrozen continuation;
- matched random-backbone control.

Record relative L2 movement by embedding, attention/mixer, FFN, normalization, and action-head
groups. A lower loss or selector score without native closed-loop improvement is not enough to
promote the weight transfer. The current AgentNet and ToolSandbox receipts therefore support
compatibility and a low-rate initialization policy, but not a claim of stateful computer-use
capability.

The m93–m96 Mind2Web→MCPMark matched triad is an explicit cross-domain reuse audit.  Frozen
transfer improves exact MCP tool selection by 5.02 percentage points over random but reduces
service routing by 20.92 points; low-rate unfreezing recovers routing to 48.12% but falls below
random on selector top-1/top-3 and moves the backbone by 0.297% relative L2.  Until a native,
verifier-backed MCP run reproduces a benefit, keep this backbone as a browser candidate only and
do not describe it as a general MCP router.

The m100 public tri-domain continuation extends that decision beyond MCP routing.  A shared
low-rate update improves the large mobile holdout but slightly degrades the desktop and tiny
browser holdouts, while moving the backbone by less than 0.5% relative L2 in every substantial
group and leaving action heads fixed.  The current policy is therefore to reuse the BPE parent
for compatibility, keep surface-specific adapters/heads, and require a matched random control
plus native closed-loop evidence before merging one universal mobile/browser/desktop adapter.

The m101–m102 matched control now confirms that the parent is not merely a convenient starting
point: it beats a deterministic random backbone on all three held-out surfaces after the same
low-rate update (`+30.76` aggregate points, with `+27.09` mobile, `+47.11` desktop, and `+60.00`
browser).  Because the warm arm still loses a little desktop/browser accuracy relative to its own
pre-update baseline, this validates pretrained initialization but not a single universal adapter;
native closed-loop and action-head ablations remain the promotion gate.

The m103–m105 frozen-backbone dispatch-head triad closes the immediate action-head gap.  With
identical 200-step cached-feature updates, warm and random heads both reach 100% route accuracy
on mobile and desktop; desktop selector top-1 ties at 73.68%, while warm wins the tiny browser
selector slice 100% to 33.33%.  The aggregate selector difference is only +2.88 points (74.82%
vs 71.94%), and warm route accuracy is 99.90% vs random 100%.  Since the non-head tensors remain
bitwise unchanged and the browser sample has only seven decisions, this supports surface-specific
head adaptation—not a universal tool-use head or native email/Notion/MCP capability.  The next
promotion gate is a resettable native AndroidWorld/BrowserGym/OSWorld/ToolSandbox-style run with
complete action and verifier logs.

The m106–m108 native ToolSandbox replay is the required reality check for that head result.  Both
the warm and random dispatch-head children execute the pinned simulator/verifier on five matched
multi-turn scenarios and finish at `1/5` (`20%`), with no warm success-rate advantage.  Warm is
slightly worse on the ambiguous contact-removal scenario (`0.0` vs `0.333` milestone similarity).
Because this remains a bounded scripted-user run rather than the official split and model-based
user simulator, it is negative evidence against promoting head transfer—not a claim of native
ToolSandbox competence.  Keep the native receipt in the publication packet and retain the
official-split requirement as a blocker.

The m109 export closes the artifact-side part of the WebGPU handoff for the warm child.  The
10.52M BPE checkpoint, tokenizer, fp32/fp16 full and hidden-only graphs, and serialized dispatch
heads all pass the exporter parity and clean static-app hash gates.  This is the bundle a hardware
browser session should exercise next; it is not itself a hardware-throughput or public Hub/Space
publication result.

The follow-up [`m110 browser probe`](paper/results/raw/m110-webgpu-warm-head-browser-probe-v1.json)
did exercise that bundle in the Codex in-app Browser.  It loaded the WebGPU-labeled path, but the
warm head selected `computer_use.click` for all four email/Notion/browser prompts (`0/4` exact).
That failure is more decision-relevant than the artifact hash: it blocks publishing this bundle
until the route/selector/action-head contract is repaired and rerun with the same prompts and a
matched control.

That rerun is now bound by the [`m111 deployment repair receipt`](paper/results/raw/m111-deployment-dispatch-repair-v1.json).  It confirms that frozen-backbone route/selector adaptation plus a small
transparent grounding/planner adapter restores the deployment contract for email, Notion, URL open,
and the first two search→Notion steps.  The matched random arm is comparable or better on the
offline mixed selector probe, so the result is a practical deployment repair—not a claim that
pretrained weights universally improve tool selection.  Native AndroidWorld, OSWorld, AgentNet,
MCPMark, EnterpriseOps-Gym, and official ToolSandbox evidence remain separate publication gates.

The [`m112 AgentNet surface-selector receipt`](paper/results/raw/m112-agentnet-surface-selector-repair-v1.json)
closes a naming mismatch discovered in the desktop projection: AgentNet's low-level `agentnet_*`
candidate surface now has its own selector in the checkpoint while the browser selector remains
intact.  Warm selector top-1 is 70.68% versus 71.43% for the matched random selector, and
end-to-end first-action type reaches 1.0, but all eight projected trajectories remain unsuccessful
because screenshots and coordinate/text grounding were deliberately not consumed.  This is a useful
negative transfer result: surface-specific selector routing helps the action-family contract, while
pretrained initialization is not a universal advantage and does not replace a native desktop adapter.

The [`m114 EnterpriseOps-Gym email receipt`](paper/results/raw/m114-enterpriseopsgym-email-retrieval-m111-ablation-v1.json)
adds an out-of-domain productivity check using 67 public email rows and 15-tool distractor pools.
The warm m111 head reaches hit@1/3/5 of 20.90/59.70/86.57%, versus 13.43/47.76/76.12% for the
matched random control.  Warm transfer therefore helps retrieval relative to random on this
surface, but it trails the older m14 diagnostic at hit@1 (26.87%) and does not establish database
verifier, MCP execution, or real-email success.

The [`m115 xLAM-derived receipt`](paper/results/raw/m115-xlam-derived-function-calling-transfer-v1.json)
adds generic function-calling coverage without claiming access to the gated Salesforce source.  On
128 rows from a public Apache-2.0 derivative test shard, row-local retrieval produces 50% first-tool
exactness and 100% schema-valid calls, but 0% exact arguments.  The global selector transfers poorly
to the unseen xLAM tool vocabulary (0.78% warm first-tool exact versus 0% random).  This separates
three failure modes for the WebGPU model: retrieval can locate a tool, schema-constrained emission
can be valid, and argument copying/multi-call planning still fail.  The original xLAM dataset is
gated here, so this is derivative evidence rather than an official Salesforce split.

The [`m116 MCPMark routing receipt`](paper/results/raw/m116-mcpmark-routing-transfer-v1.json)
extends the realistic tool surface to a pinned public MCP benchmark whose tasks cover Playwright,
Notion, filesystem, GitHub, and Postgres workflows.  Across all 169 standard task descriptions,
the warm and random checkpoints both route 14.79% correctly; on the 70 easy descriptions both are
14.29%.  The model recognizes the Playwright family (100% in both suites) but routes 0% of the
Notion, filesystem, GitHub, and Postgres families.  This is actionable transfer evidence for the
WebGPU deployment: browser primitives are learned, while service-specific tool naming and
multi-step state planning are not.  MCPMark's actual MCP servers and verifiers were not started,
so the receipt is a task-description proxy rather than task success.

The [`m117 trajectory provenance receipt`](paper/results/raw/m117-mcpmark-trajectory-metadata-v1.json)
confirms that the public [MCPMark trajectory-log dataset](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log)
contains multi-turn, tool-result-grounded traces under an MIT license.  A bounded filesystem trace
has 45 events, 21 paired calls, and five concrete tools (`list_directory`, `read_text_file`,
`read_multiple_files`, `move_file`, and `list_allowed_directories`).  Because the trace includes
large third-party document outputs, this pass deliberately stores only provenance and counts; it
does not silently turn public logs into training data.  A content/license audit and argument/schema
normalizer are required before any SFT or distillation use.

The [`m118 redacted trajectory SFT receipt`](paper/results/raw/m118-mcpmark-redacted-trajectory-sft-transfer-v1.json)
implements that audit on three hash-pinned public traces (filesystem, Notion, and Playwright; 67
paired calls).  The normalizer keeps user prompts and structured arguments, but replaces every tool
result and assistant free-text response with fixed markers and rewrites absolute paths to workspace
suffixes.  Filesystem and Notion rows train a 16-step teacher-forced continuation; the Playwright
row is held out.  Both warm and matched-random arms have identical train/eval loss and identical
backbone movement, with held-out loss improving from 5.6167 to 5.5599 but sequence accuracy staying
at 0%; a one-step unseen Playwright probe selects the wrong action in all four arms.  This is useful
evidence that redacted multi-turn traces are ingestible and that the current tiny model still lacks
stateful action planning.  It is not an official MCPMark score, native MCP execution, or a reason to
publish a universal adapter.

The [`m119 dynamic-selector receipt`](paper/results/raw/m119-mcpmark-dynamic-selector-transfer-v1.json)
then separates representation transfer from teacher-forced language loss.  A 22-tool global schema
catalog is built from the same redacted rows, and a frozen-feature dense selector is trained only on
the 51 filesystem/Notion assistant decisions.  On the 19-call held-out Playwright trajectory, the
warm WebGPU body reaches top-10 routing of 84.21% while the matched random-body control reaches 0%;
both top-1 scores are 0%.  The warm arm therefore carries useful broad retrieval signal into the
unseen browser catalog, but cannot make the exact first choice.  This supports retaining the
pretrained body for candidate retrieval and adding service/action adapters; it does not justify
claiming native MCP success or a universal tool policy.

The [`m120 broad redacted SFT receipt`](paper/results/raw/m120-mcpmark-broad-redacted-sft-transfer-v1.json)
expands the same public trajectory method across five MCP services: filesystem, Notion, GitHub,
Postgres, and Playwright.  Eight rows (107 calls) train and two Playwright rows (24 calls) remain
held out.  After the same 32 CPU SFT steps, the warm 10.52M body reaches held-out token accuracy
of 42.27% versus 3.12% for a matched random body, a +39.15-point transfer gap; exact sequence
accuracy is still 0% in both arms.  The warm body moves only 0.345% aggregate relative L2
(embedding 0.466%, mixer 0.181%, FFN 0.212%), while the random body moves 102.06% aggregate.
This is stronger evidence for reusing pretrained representations and assigning larger rates to
new heads, but it remains teacher-forced, redacted, and non-native.

### MCPMark state-summary trajectory transfer (m213)

The [`m213 receipt`](paper/results/raw/m213-mcpmark-state-summary-transfer-v1.json) tests whether
the fixed tool-result marker was hiding the state signal needed for recovery.  A new
[`normalize_mcpmark_state_trajectory.py`](../scripts/normalize_mcpmark_state_trajectory.py) keeps
only a deterministic result status (`ok`/`error`), coarse shape/content types, bounded character
counts, a page-state boolean, and a short digest.  It retains no result text, URLs, document
content, identifiers, or assistant free text.  The split is whole-trajectory disjoint: eight
filesystem/Notion/GitHub/Postgres rows train and two Playwright rows remain held out.

After 64 matched CPU continuation updates, the warm 10.52M body rises from `38.01%` to `41.46%`
held-out assistant-token accuracy, versus `0.79%` to `15.63%` for the random body.  Warm movement
is small (embedding `0.887%`, mixer `0.319%`, FFN `0.374%`, normalization `0.013%`) and action
heads remain unchanged; exact sequence accuracy is `0%` for both.  A frozen global selector still
gets `0/24` top-1 decisions on the unseen Playwright service for both arms.  This makes the state
summary a better provenance-safe training primitive than a fixed marker, but not yet a transferable
MCP policy.  The children are not exported or adopted, and no MCP server, browser, verifier, or
official split was executed.

### Direct m180 WebGPU child comparison (m193)

The [`m193 receipt`](paper/results/raw/m193-current-m180-webgpu-browser-fresh-realistic-actions-v1.json)
closes a provenance gap by exercising the separately exported `108276…` m180 child itself rather
than the workspace's older `9bba…` bundle.  The browser reports `WEBGPU` and model-ready status,
but the same twelve realistic prompts used for the workspace smoke produce only `1/9` exact tools
on unambiguous single-step requests and `0/2` exact planner trajectories.  The sole exact tool is
an explicit URL open; email and Notion requests still select `open_url` with malformed URL-shaped
arguments.  The run has no external side effects and uses no public benchmark rows.

This comparison is the clearest current separation between transfer and deployment quality:
teacher-forced public continuations can improve token accuracy and route loss while the actual
WebGPU action contract remains unusable for productivity and multi-step control.  The policy is to
retain the m180 body for compatibility experiments, but not promote it as a browser/mobile agent
until a catalog-aware action head, argument grounding, and native closed-loop evidence all improve
together.

### Native WebGPU capability and latency receipt (m214)

The [`m214 receipt`](paper/results/raw/m214-webgpu-native-capability-current-bundle-v1.json) is the
first current-workspace browser receipt that proves the requested provider path rather than merely
loading a page.  Chromium reports `navigator.gpu`, exposes an Apple Metal adapter
(`vendor=apple; architecture=metal-3`), and ONNX Runtime Web `1.27.0` runs the fp16 action graph
with `requested_provider=["webgpu"]` and no provider retry.  The clean eight-artifact bundle and
its parity manifest verify before dispatch.

Across three bounded local prompts and 30 measured repetitions per prompt, p50 latency is `7.2 ms`
and the action-input throughput proxy is about `1,389 tokens/s`; the conservative graph-plus-host-
tensor estimate is `20.5 MB`.  Quality is separate: URL opening is exact in `30/30` trials, while
the email request selects `email_send` instead of the expected `send_email` and the Notion request
selects `notion_create_page` instead of `notion_write`.  Thus native WebGPU capability and speed
pass, but realistic email/Notion action quality is `1/3` and closed-loop success is zero.  No real
email, browser navigation, or Notion account was touched, and this single Apple Metal run is not a
cross-device throughput claim or a publication-quality agent score.

The [`m215 gate`](paper/results/raw/m215-workshop-gate-native-webgpu-v1.json) confirms the scope of
this result: the WebGPU capability/latency requirement and the existing matched transfer ablation
now pass the strict checker.  Readiness is still false with twelve blockers—eleven official native
benchmark receipts plus a public model/demo manifest.  The local browser result cannot substitute
for AndroidWorld, MobileGym, iOSWorld, BrowserGym, OSWorld, AgentNet, ToolSandbox, MCPMark, or
EnterpriseOps-Gym execution.

### Canonical productivity schema boundary (m216)

The [`m216 receipt`](paper/results/raw/m216-webgpu-native-canonical-productivity-v1.json) reruns
the same native Chromium harness after fixing a duplicate-tool precedence bug in the demo runtime.
The bundle metadata contains both the public `send_email`/`notion_write` schemas and legacy
`email_send`/`notion_create_page` mobile aliases.  The guard now prefers the public names and falls
back to an alias only when a canonical entry is absent, so argument grounding uses `recipient` and
`content` rather than the legacy `to`/`subject`/`body` or page shape.

On the same Apple Metal adapter, all three bounded prompts are now schema-exact (`3/3`, 30/30 per
case), with p50 latency `5.1 ms` and an action-input throughput proxy of about `1,918 tokens/s`.
This is a contract/adapter correction, not a learned-quality improvement: email and Notion still
use explicit intent guards, no external account is touched, and closed-loop success remains zero.
The strict workshop gate therefore remains unchanged; native benchmark receipts and a public
model/demo manifest are still required.

The [`m217 gate`](paper/results/raw/m217-workshop-gate-native-canonical-v1.json) binds this
canonical receipt directly: the WebGPU and m25 weight-ablation checks pass, but the gate still has
the same twelve blockers (eleven official native benchmark receipts and the public model/demo
manifest).  The alias fix therefore improves contract observability without changing the
publication decision.

### Current native evidence join (m218)

The [`m218 gate`](paper/results/raw/m218-workshop-gate-current-canonical-native-v1.json) binds
the canonical m216 WebGPU receipt to two existing official-split native evaluations: MobileGym
(`256` tasks, `13/256` success) and BrowserGym/MiniWoW (`240` episodes, `0/240` success).  It also
binds the live public model/demo manifest and the m25 transfer/no-transfer ablation.  The gate
now has nine blockers rather than twelve.  ToolSandbox is deliberately still blocked because its
129-scenario audit does not verify the official split; this prevents a local simulator result from
being promoted to a benchmark claim.

### ToolACE public source and transfer audit (m219)

The [`m219 receipt`](paper/results/raw/m219-toolace-public-source-transfer-audit-v1.json) adds
Team-ACE/ToolACE to the supplemental source registry.  The pinned Apache-2.0 snapshot contains
`11,300` raw rows; the strict projection accepts `8,993`, with `8,044` train parent records and
`949` eval parent records, zero parent overlap, and zero prompt overlap.  Rows that lack a strict
first action or valid schema are rejected rather than silently converted into supervision.

Three matched current-child transfer receipts cover first-action, full multi-turn, and
action-history projections.  Warm-start token accuracy beats the matched random backbone by
`17.37–49.48` percentage points, while warm sequence exactness remains `0%` in every arm.  The
adoption conclusion is therefore narrow and reproducible: keep the pretrained body for further
ToolACE continuation, but do not promote the tool heads or call the result native MCP, BFCL,
mobile, browser, desktop, email, or Notion capability.

### Current AgentNet text-action evaluation and matched control (m220)

The [`m220 receipt`](paper/results/raw/m220-agentnet-current-text-action-evaluation-v1.json) runs
the current m180 WebGPU-compatible child and a matched random-backbone control through the pinned
public [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) text projection.  The held-out
projection contains `133` action rows from `8` unseen Ubuntu parent trajectories, with screenshots
and desktop state deliberately excluded.  Both arms cover every expected parent ID, so the score is
not a missing-prediction artifact.

The warm arm predicts the correct first action *type* on all eight parents (`100%` versus `12.5%`
for random, `+87.5` percentage points), but both arms remain at `0%` exact trajectories and
effectively zero coordinate/text action score.  The existing tensor audit finds `51` compatible
shared tensors, tokenizer equality, and warm embedding/attention/FFN relative movement of only
`0.449%`/`0.223%`/`0.273%`.  This supports retaining the pretrained body as an initialization
candidate, not adopting it as a computer-use solution: visual grounding plus native AgentNetBench
or desktop execution is still required.

### AgentNet public continuation and matched random transfer (m221)

The [`m221 receipt`](paper/results/raw/m221-agentnet-public-continuation-transfer-v1.json) is the
first current-child SFT continuation on the pinned AgentNet text/action projection rather than a
selector-only probe.  It uses `513` train rows from `32` parents and `133` held-out rows from `8`
unseen parents, with the same `32`-step schedule for the warm m180 checkpoint and a matched random
backbone.  The split is parent-disjoint and screenshots remain excluded.

Warm reuse improves held-out teacher-forced token accuracy from `55.80%` to `67.00%` and route
accuracy from `18.80%` to `83.46%`; the random control reaches `21.22%` token accuracy and `0%`
route accuracy.  The stronger action evaluator is decisive: warm and random children both remain
at `0%` exact trajectories and `0` meaningful coordinate/text action score.  The tensor audit
finds `51` compatible tensors, equal tokenizers, and warm embedding/attention/FFN movement of
`0.532%`/`0.284%`/`0.346%`.  The adoption decision is therefore to keep warm reuse for a future
visual/action-grounded continuation, while refusing to promote this text-only training result to
native desktop or AgentNetBench capability.

### AgentNet continuation to local productivity bridge (m222)

The [`m222 receipt`](paper/results/raw/m222-agentnet-continuation-stateful-productivity-bridge-v1.json)
checks whether the m221 computer-use continuation preserves the requested email, Notion, browser,
and recovery contract.  The resettable in-memory fixture has `63` tools and `5` workflows; its
oracle reaches `16/16` accepted steps and `5/5` complete tasks, confirming the verifier itself.

The warm child accepts `5/16` model steps versus `1/16` for the matched random child, but both
complete only the abstention workflow (`1/5`).  Warm email and Notion completion remain `0/1`, as
do browser and recovery completion.  This is a useful cross-surface regression signal—the warm
backbone preserves more valid transitions—but it is deliberately not presented as real email,
Notion, MCP, browser, or WebGPU control.

### AgentNet continuation to MCPMark service-routing bridge (m223)

The [`m223 receipt`](paper/results/raw/m223-agentnet-continuation-mcpmark-routing-bridge-v1.json)
reruns the current m221 warm child and matched random control on the pinned public
[MCPMark](https://github.com/eval-sys/mcpmark) standard (169 rows) and easy (70 rows) task-description
manifests.  This is the closest available public proxy for the requested Notion, filesystem,
GitHub, Postgres, and browser tool families while the native MCP services are unavailable.

Warm routing reaches `19.53%` on standard and `22.86%` on easy, versus `22.49%` and `18.57%`
for random.  The warm standard breakdown is informative rather than strong: Notion is `10/28`
(`35.71%`) and Playwright is `23/25` (`92%`), while filesystem, GitHub, and Postgres are `0/30`,
`0/23`, and `0/63`.  Selector top-1 remains `0%` for the warm arm.  The mixed result supports
retaining a compatible warm body for service-grounded continuation, but does not justify native
MCP, real-account email/Notion, or WebGPU productivity claims.  MCP servers, credentials, state
transitions, verifiers, and pass@k were not run, and the task text was not retained for training.

### Frozen dispatch repair control (m194)

The [`m194 receipt`](paper/results/raw/m194-current-m180-dispatch-repair-v1.json) tests whether the
m180 child can be repaired without changing its WebGPU-compatible backbone.  A matched warm-start
and random-head arm trains route and dense-selector heads on 741 source-disjoint AgentNet/Mind2Web
projection rows plus 1,286 deterministic local adapter rows; 320 additional synthetic adapter rows
are held out.  Warm routing reaches 99.61% versus 99.42% for random, but warm selector top-1 is
82.75% versus 82.95%, and canonical tool probes are 4/6 versus 5/6.  The warm head therefore is
not adopted and no export is requested.  This is evidence for measuring head transfer separately
from backbone reuse, not evidence of native browser, mobile, MCP, visual, or real-account control.

### Unguarded m194 WebGPU comparison (m195)

The [`m195 receipt`](paper/results/raw/m195-current-m194-webgpu-browser-unguarded-v1.json) exports
the m194 warm head with hard PyTorch↔ONNX parity and runs the exact prompt set used by m193 through
the unguarded comparison shell.  After explicit catalog aliases (`notion_write`→`notion_create_page`
and `send_email`→`email_send`), tool-family routing rises from the m180 parent's `1/9` to `5/9`.
Strict action exactness stays at `1/9`, however: URL opening is correct, while search, click, email,
Notion, and calendar arguments remain incomplete or contain tokenizer markers.  The search→Notion
planner emits the right two tool families but still fails strict arguments, and the file/test/commit
planner abstains (`0/2` complete trajectories).  This isolates the next required work: argument
copy/grounding and state-conditioned planning, not another route-head-only update.  The temporary
export is therefore not promoted or published.

### Public DOM grounding control (m196)

The [`m196 receipt`](paper/results/raw/m196-m194-grounded-mind2web-vocab-fix-v1.json) also fixes a
pipeline bug exposed by current checkpoints: m194 carries a 23-row stateful pointer vocabulary,
while the older Mind2Web trainer assumed 17 legacy rows.  The migration now copies embeddings by
declared argument name and appends `target_id`/`value`, yielding a 25-row compatible head.  A
32-step continuation over 219 public train decisions and 63 disjoint eval decisions still gives
`0/63` exact pointer spans before and after.  The body moves only 0.274%/0.422%/0.507% in
embedding/mixer/FFN groups, while pointer and tool heads move more.  The result confirms that
argument grounding needs aligned span supervision and an explicit DOM/action contract; more
route-head training will not solve it.  The child is not exported or adopted.

### Frozen-feature pointer transfer (m197)

The [`m197 receipt`](paper/results/raw/m197-pointer-grounding-repair-v1.json) isolates the
argument head with the m194 backbone completely frozen.  It caches token-position features for
219 public Mind2Web train spans and trains matched warm and random start/end pointer heads for 400
steps.  Warm transfer reaches `3/63` held-out exact spans (4.76%) while random reaches `5/63`
(7.94%).  The warm head moves substantially less (`31.42%` shared pointer L2 versus `134.06%`),
but the lower movement does not carry useful unseen-value grounding.  This rules out adopting the
warm pointer solely because it preserves pretrained weights; the next adapter must align DOM
candidate serialization, action argument schemas, and span labels before another WebGPU export.

### Current native ToolSandbox smoke (m198)

The [`m198 receipt`](paper/results/raw/m198-toolsandbox-native-current-v1.json) runs the current
m194 checkpoint in the pinned Apple [ToolSandbox](https://github.com/apple/ToolSandbox) simulator
and milestone verifier.  It completes `3/3` resettable single-turn scenarios—cellular-off,
Wi-Fi-off, and send-message—with mean milestone similarity `1.0`; no external API is called.
This is the first current-checkpoint stateful-tool result for the deployment path, but it remains a
diagnostic smoke: the scripted user ends after the first response, multi-tool and multi-turn cases
are truncated, and the official split/model-based user simulator/full scenario matrix were not
executed.  It therefore does not establish native MCP, email, Notion, or publication-gate success.

The follow-up [`m200 comparison receipt`](paper/results/raw/m200-toolsandbox-native-current-base-v1.json)
runs all 129 base/no-distraction scenario names used by the prior m143 transfer audit with the
m194 checkpoint.  The current child reproduces the baseline exactly: `28/129` success (`21.71%`),
mean similarity `0.2921`, and zero per-scenario wins or losses.  This is a useful negative result:
the m194 route/selector repair did not change native ToolSandbox behavior under the one-step
protocol, so it is not promoted.  The same official-split/user-simulator boundary still applies.

The [`m201 interactive receipt`](paper/results/raw/m201-toolsandbox-native-current-interactive-v1.json)
extends the check to five stateful, ambiguous, and multi-turn scenarios with the bounded interactive
scripted user.  The m194 child reaches `1/5` (`20%`) and exactly matches the prior m92 stateful,
public-only, and projection arms on every scenario (`0.25`, `0.3333`, `1.0`, `0.0`, `0.5`).  The
model therefore still cannot turn route repair into reliable multi-tool state progression; the
next training target is explicit state/action-history supervision and recovery, not another frozen
selector update.

### Public ToolSandbox continuation and native bridge (m202–m204)

The [`m204 receipt`](paper/results/raw/m204-toolsandbox-continuation-native-bridge-v1.json) runs the
AST-only public projection from the pinned [ToolSandbox](https://github.com/apple/ToolSandbox) source:
107 train rows and 20 source-disjoint eval rows.  A matched 32-step warm continuation improves
held-out assistant-token accuracy from `65.79%` to `73.21%`, while the random-backbone control
reaches only `8.61%`; warm embedding/mixer/FFN movement stays at `0.52%/0.19%/0.24%`, and the
action heads do not move.  This is useful evidence that the pretrained body transfers public
ToolSandbox language signal, but the native bridge is decisive: warm and random children both
score `1/5` with identical per-scenario similarities in the interactive simulator.  The children
are therefore not exported or adopted; action-head and state-history supervision must be trained
explicitly before another WebGPU release.

### ToolSandbox selector repair and native bridge (m205–m207)

The [`m207 receipt`](paper/results/raw/m207-toolsandbox-selector-repair-bridge-v1.json) isolates
the next hypothesis: keep the m194 backbone, replace only the dense selector, and train against the
actual 32-tool catalog extracted from the public ToolSandbox AST projection.  On the same 107 train /
20 eval rows, held-out candidate top-1 rises from 0% to 50% for the warm arm and 45% for the matched
random arm (top-3 reaches 65% for both).  Selector movement is large (warm query/tool towers
1.12/0.58 relative L2; random 1.25/1.06), so this is a genuine head-only adaptation rather than
backbone drift.

The native bridge rejects that offline gain: both children score exactly `1/5` on the five-scenario
interactive simulator/verifier stress set, with identical per-scenario similarities
(`0.25`, `0.3333`, `1.0`, `0.0`, `0.5`).  The AST row-local candidates do not match the runtime
ToolSpec serialization and state/action-history conditions used by `run_toolsandbox_native.py`;
the route and pointer heads also remain inherited.  Therefore neither child is exported or adopted.
The next experiment must bind the training rows to the same runtime candidate serializer and
state-history format, then train route, selector, and argument/action heads jointly with native
closed-loop replay.  This result is diagnostic transfer evidence, not an official ToolSandbox score
or WebGPU productivity claim.

### ToolSandbox runtime ToolSpec alignment and native bridge (m212)

The [`m212 receipt`](paper/results/raw/m212-toolsandbox-runtime-heads-native-bridge-v1.json) closes
the candidate-catalog mismatch without overstating the result.  A pinned Apple ToolSandbox checkout
enumerates 1,032 named scenarios; the projection calls the same
`ExecutionContext.get_available_tools(scrambling_allowed=False)` path used by the native runner and
serializes the exact OpenAI-style ToolSpecs.  It retains 105 train and 20 eval rows, dropping two
static rows whose target is intentionally unavailable in the runtime context.  The projection still
has no simulator state or action history, and no upstream user simulator or official split is run.

Joint frozen-backbone route/selector training raises the warm row-local selector from `5%` to `70%`
top-1 (`100%` top-3) and route-to-`app_action` from `5%` to `100%`.  The matched random control reaches
`80%` selector top-1 and the same `100%` route score, so the offline result does not demonstrate
pretrained-weight advantage.  More importantly, the corrected native bridge now passes the loaded
selector into decoding, yet warm and random children both remain at `1/5` (`20%`) native success.
The warm child is therefore not exported or adopted.  The remaining blocker is explicit state/action
history plus argument/action-head learning under the simulator's milestone verifier, not another
catalog-only probe.

### Public Space black-box regression (m224)

The [`m224 receipt`](paper/results/raw/m224-public-space-black-box-realistic-prompts-v1.json)
records a fresh in-app-browser inspection of the public
[`danelcsb/localagent-webgpu`](https://huggingface.co/spaces/danelcsb/localagent-webgpu) Space.
The page is live (`HTTP 200`, `Running`, `WEBGPU`) but serves an older 10,986-byte `app.js`; a
manifest probe returns `Entry not found`, so this is not the current 10.52M BPE workspace bundle.
The explicit URL prompt selects `open_url` (`1/1`, 47 ms).  The realistic email prompt selects
`set_reminder` instead of `send_email`/`email_send` (`0/1`, 37 ms), and the compound Search→Notion
prompt emits one `notion_write` proxy without a verified search result, state transition, or final
write (`0` closed-loop completions).  No credentials or external side effects were used.  The
result is retained as a public regression and publication-boundary check, not promoted to browser,
email, Notion, MCP, or official benchmark evidence; updating the Space requires authenticated HF
upload followed by public hash and behavior re-verification.

### Current EnterpriseOps-Gym email retrieval (m225)

The [`m225 receipt`](paper/results/raw/m225-enterpriseopsgym-current-checkpoint-email-retrieval-v1.json)
evaluates the current 10.52M-parameter m194 warm dispatch-repair checkpoint on the pinned public
EnterpriseOps-Gym email slice: 67 oracle rows paired with the same 67 `plus_15_tools` candidates
used by m114.  Name-only retrieval reaches Hit@1/3/5 of `49.25%/85.07%/95.52%`, compared with
`20.90%/59.70%/86.57%` for m114 (`+28.36/+25.37/+8.96` percentage points).  The repair froze
the backbone (`0%` relative movement) and moved the dense selector by `148.87%`, identifying a
head-only contract adaptation rather than evidence that the pretrained body is superior.
Because verifiers, server configuration, and execution were removed, the result is useful for
email-tool retrieval but cannot be promoted to EnterpriseOps-Gym, MCP, real-email, or WebGPU
closed-loop success.

### WebGPU side-effect and injection boundary (m227)

The [`m227 receipt`](paper/results/raw/m227-webgpu-side-effect-safety-policy-v1.json) adds an
explicit runtime policy around the model's structured action output.  `open_url` is read-only and
allowed; email, Notion, messaging, file, shell, and other state-changing tools are marked
`confirmation_required`; destructive actions receive high severity; and prompt-injection or
secret-exfiltration indicators in the request or untrusted observation are blocked.  The seven-case
audit is pure JavaScript (`side_effect_confirmation_v1`), changes no learned weights, and executes
no external effects.  It is a deployment safety contract, not a VPI-Bench, AgentCIBench,
MobileSafetyBench, or native task-success score.

### Current public-release reconciliation (m228)

The [`m228 receipt`](paper/results/raw/m228-current-public-release-audit-v1.json) rechecked the
authoritative public pages before extending the evaluation plan.  BFCL V4 now documents separate
agentic web-search, memory-management, multi-turn, and format-sensitivity categories in addition
to the older AST call suites; these are evaluation-only because they require the upstream checker,
live search, or stateful memory runtime.  iOSWorld's public release explicitly includes 26 apps,
133 personalized/cross-app tasks, and an optional MCP server, so it is a native identity/state
benchmark rather than a screenshot-only action set.  OSWorld-V2 requires the exact
`osworld-v2-2026.06.24` release across code, gated task classes/assets, mocked websites, and
provider image; floating `main` is not comparable.

The audit also found a release-scope discrepancy that must remain visible: the current
[MobileSafetyBench project page](https://mobilesafetybench.github.io/) describes 250 tasks (200
daily scenarios plus 50 injection scenarios), while the pinned paper/repository contract used by
the earlier m178/m179 receipts describes a 100-task suite (50 helpfulness, 42 safety, 8 injection).
Those counts are not additive or interchangeable.  Before a native safety score, one release must
be selected and its task manifest, APKs, emulator, and verifier hash-bound.  No m228 rows entered
training and no runtime was executed.

### Stateful productivity and user-in-the-loop refresh (m229)

The [`m229 receipt`](paper/results/raw/m229-stateful-productivity-benchmark-refresh-v1.json)
updates the contract for the workflows the WebGPU demo is meant to approach.  MCPMark's public
task format requires `meta.json`, `description.md`, and an independent `verify.py` state check over
Notion, GitHub, filesystem, Postgres, and Playwright services; its documented tasks average 16.2
execution turns and 17.4 tool calls, so a single correct tool name is not task completion.
EnterpriseOps-Gym exposes 649 resettable enterprise tasks, including 67 email rows, with selected
tools, MCP server state, and SQL/database-state verifiers.  These are the closest public email and
Notion-adjacent stateful contracts in the current matrix, but no server or account is attached here.

AppWorld-UL adds the missing user-interaction dimension: 516 tasks require clarification,
confirmation, or handling an infeasible instruction.  Its official page links the public
[Apache-2.0 AppWorld runner](https://github.com/StonyBrookNLP/appworld), whose documented
installation (`pip install appworld`, `appworld install`, `appworld download data`) and resettable
`AppWorld(...).evaluate()` loop establish an executable base contract.  The UL-specific task assets
and knowledge-bounded user-simulator release are not separately pinned on that page, so this is
runner evidence rather than a native UL score.  This maps directly to the deployment policy:
the model must distinguish an allowed read, a confirmation-required write, a clarification request,
and a blocked action before calling a tool.  τ³-Bench adds knowledge-grounded policy/tool use and
full-duplex voice, while TUA-Bench covers 120 general terminal tasks including email management and
live-web information seeking; both remain evaluation-only for this text-first WebGPU model.

The adoption conclusion is unchanged but sharper: warm-start token or retrieval gains are useful
initialization evidence only.  Promotion requires matched random/no-transfer arms, per-group tensor
movement, and native state-verifier success on the same pinned release.  m229 adds no training rows,
does not execute external effects, and does not convert the local email/Notion state machine into an
official MCPMark, EnterpriseOps-Gym, or AppWorld-UL score.

### Structured-action safety boundary fix (m230)

The [`m230 receipt`](paper/results/raw/m230-webgpu-structured-action-safety-fix-v1.json) caught a
real deployment-path mismatch: planner calls supplied a tool name as a string, while the ordinary
UI path supplied `{tool, args}`.  The policy reader only inspected `{name}`, so the second path
could have been labeled read-only even for `send_email` or `notion_write`.  The policy now accepts
all three action shapes (`name`, `tool`, and nested `action.tool`), and the regression covers
structured email/Notion confirmation, read-only URL allowance, and injection blocking for an
interactive click.  This fixes policy observability; it is not a learned-quality or native task
success result.

### User-in-the-loop clarification boundary (m231)

The [`m231 receipt`](paper/results/raw/m231-webgpu-user-in-the-loop-clarification-policy-v1.json)
adds the missing AppWorld-UL interaction state to the static WebGPU path.  A structured
`send_email` action with no `recipient` is now `clarification_required`; the same action with a
recipient is `confirmation_required`; a complete `open_url` is `allowed`; and an injected Notion
write is `blocked`.  The policy is schema-aware but remains a deterministic deployment guard, not
learned capability or AppWorld-UL benchmark performance.

### Stateful productivity GRPO child and weight preservation (m234–m235)

The [`m234 receipt`](paper/results/raw/m234-stateful-productivity-grpo-head-preserved-v1.json)
is a fresh pure-PyTorch continuation from the current 10.52M Mind2Web-adapted BPE child.  It
builds 16 source-disjoint local training decisions over email, Notion, browser search/recovery,
and abstention, runs a 32-update SFT prelude plus four GRPO rollout updates, and preserves the
five deployment-head containers (`tool_head`, `ptr_head`, `route_head`, `dense_selector`, and
`selector_proj`) as frozen tensors in the child checkpoint.  Mean shaped reward improves from
`0.0375` to `0.1125`, but held-out exact tool/text accuracy remains `0%`; the result is therefore
an RL-pipeline and reward-signal diagnostic, not a capability claim.

The [`m235 weight audit`](paper/results/raw/m235-stateful-productivity-rl-weight-transfer-v1.json)
confirms identical model configuration, shapes, and tokenizer identity.  Frozen action heads move
by `0%` relative ΔL2; the LM embedding moves `4.69%`, attention/mixer `1.40%`, FFN `1.74%`, and
normalization `0.10%`.  This makes the adoption decision explicit: retain the compatible backbone
and frozen heads only as a lineage-preserving checkpoint, and do not promote it to WebGPU or an
official benchmark until a matched no-transfer arm improves held-out exactness and a native state
verifier succeeds.  The earlier run that silently dropped auxiliary heads is superseded by this
head-preserving path and remains unadopted.

The [`m236 deployment smoke`](paper/results/raw/m236-stateful-productivity-rl-deployment-smoke-v1.json)
confirms that the preserved-head child is loadable by the actual LocalAgent runtime, but its
10-case echo-stub dispatch is only `1/10` exact.  It therefore regresses the older local parent on
this smoke and is explicitly not exported or promoted; preserving tensors fixes checkpoint
integrity, not policy quality.

### Matched random RL control (m237–m240)

The [`m237 random-control receipt`](paper/results/raw/m237-stateful-productivity-grpo-random-control-v1.json)
repeats the m234 protocol with a shape-matched random backbone and random auxiliary heads.  The
train/eval task hashes, seed, 32-step SFT prelude, four GRPO steps, rollout budget, and frozen-head
contract are identical.  The random arm stays at mean reward `−0.05`, produces zero informative
groups and zero realized optimizer updates, and has `0%` held-out exactness.

The [`m240 matched ablation`](paper/results/raw/m240-stateful-productivity-grpo-matched-ablation-v1.json)
joins both arms with tensor movement and deployment smoke.  Warm initialization has a shaped-reward
advantage of `+0.1625`, but exact tool/text accuracy ties at `0%`, while the random child wins the
10-case echo-stub deployment smoke `2/10` to `1/10`.  The decision is therefore explicit:
pretrained weights help the local reward signal but are not adopted for WebGPU deployment.  This
control is still a deterministic local simulation, not public benchmark or native environment
evidence.

### AppWorld native checkpoint probe (m241)

The [`m241 receipt`](paper/results/raw/m241-appworld-current-checkpoint-native-probe-v1.json)
adds the first resettable stateful-app runtime to this evidence chain.  An isolated AppWorld
`0.2.0.dev0` installation was unpacked from the public runner's Git-LFS bundles, its data version
`0.2.0` was downloaded, and the runner's own end-to-end verifier passed `1/1` task.  The current
10.52M BPE checkpoint was then evaluated on six public train tasks spanning SimpleNote, phone
messaging, and Venmo.  Every task reset and ran its ground-truth verifier, but native success was
`0/6`: the model emitted no AppWorld API action (two prompts were misrouted to `write_clipboard`,
four returned text), and no action was replayed.

This is stronger than a local fixture because the state database and verifier are real AppWorld
artifacts, but it is deliberately a zero-action interface baseline.  LocalAgent's compact tool
syntax still needs a schema-aware AppWorld API translator before an end-to-end agent score is
meaningful; m241 is not AppWorld-UL, email/SMS success, or a promotion signal.

### AppWorld `run_python` adapter and native replay (m242–m245)

The [`m242 train manifest`](paper/results/raw/m242-appworld-train-manifest-v1.json) and
[`m242 dev manifest`](paper/results/raw/m242-appworld-dev-manifest-v1.json) normalize 24 public
AppWorld train tasks and 12 disjoint public dev tasks into the canonical `Conversation` format.
The ground-truth programs are represented as the existing `run_python` tool; no protected test
split or raw solution text is committed.  This is a deliberately narrow adapter for AppWorld's
executable Python/API contract, not a claim that a text-only tool vocabulary is already an
AppWorld policy.

Warm SFT continuation from the current 10.52M BPE checkpoint ([`m242 report`](paper/results/raw/m242-appworld-runpython-sft-v1.json))
raises held-out assistant-token accuracy from `10.21%` to `21.60%` and lowers mean loss from
`6.645` to `5.821` on the 12 dev rows, while sequence exactness remains `0/12`.  The separate
[`m243 weight report`](paper/results/raw/m243-appworld-runpython-weight-transfer-v1.json)
shows compatible config/tokenizer/shapes, frozen action-head movement of `0%`, and small shared
backbone movement (embedding `0.57%`, attention/mixer `0.32%`, FFN `0.38%`).  These are weight-lineage
and teacher-forcing measurements, not native task success.

The [`m244 head adapter`](paper/results/raw/m244-appworld-runpython-head-adapter-v1.json)
trains only the route and dense selector for 256 steps against the exact 51-tool standard pool;
route and selector top-1 both move from `0%` to `100%` on the disjoint dev rows.  Runtime retrieval
was explicitly widened to all 51 tools because a top-10 retriever can otherwise hide
`run_python`, even when the selector ranks it correctly.

Finally, [`m245 native replay`](paper/results/raw/m245-appworld-runpython-native-dev-v1.json)
captures and executes the model's selected `run_python` call on those 12 resettable AppWorld dev
tasks.  All 12 calls were replayed, but they made `0` AppWorld API requests and passed `0/12`
verifiers.  The result is the required negative capability boundary: routing is learnable, while
the tiny model still does not emit executable API programs.  The adapter is therefore not adopted
for WebGPU or presented as an AppWorld leaderboard score; the next required step is schema-aware
program synthesis/repair plus native multi-interaction evaluation.

### AppWorld first-action API-step adapter (m247–m251)

To test whether the interface gap was only the multi-thousand-token program length, the public
train/dev ground-truth traces were reduced to their first non-bootstrap API call.  The
[`m247 train manifest`](paper/results/raw/m247-appworld-action-train-manifest-v1.json) contains 24
train tasks and the disjoint [`m247 dev manifest`](paper/results/raw/m247-appworld-action-dev-manifest-v1.json)
contains 12 dev tasks.  Credentials and bootstrap calls are excluded; the raw `Conversation`
JSONL remains outside Git and the manifests bind only task/source/action hashes.

Warm continuation ([`m247 report`](paper/results/raw/m247-appworld-action-step-sft-v1.json)) raises
held-out assistant-token accuracy from `51.38%` to `82.21%`, but sequence exactness remains `0/12`.
The standard 51-tool route and selector were already `100%` before and after training.  The paired
[`m251 weight report`](paper/results/raw/m251-appworld-action-step-weight-transfer-v1.json) shows
action heads frozen (`0%` movement), with embedding `2.12%`, attention/mixer `0.83%`, FFN `1.05%`,
and normalization `0.055%` relative movement.  These are teacher-forced and lineage measurements,
not executable-agent accuracy.

The native diagnostic adds a strict AST parser that accepts only one literal
`apis.<app>.<api>(...)` call and injects credentials from the resettable AppWorld fixture.  A
schema-grounded candidate mode ([`m250 receipt`](paper/results/raw/m250-appworld-action-step-schema-native-dev-v1.json))
replayed all 12 bounded calls and made 12 real action API requests (48 requests including 36
authentication/bootstrap requests), but the one-step verifier score was `0/12`.  The model-ranked
valid candidates were often the wrong API for the instruction; this proves isolated execution and
the parser/credential boundary, not action selection, full trajectories, AppWorld-UL, or email/SMS/
Spotify task success.  Learned free-form `run_python` code is still not used by the constrained
decoder, so this adapter is not promoted to WebGPU.

### τ-Bench native mock-domain probe (m252)

The current public [tau2-bench](https://github.com/sierra-research/tau2-bench) checkout is pinned
to `363133ada1936491fb5bcec33cd62c3518a99f65` (v1.0.1).  Its repository is MIT-licensed, but the
benchmark task/user-simulator protocol remains evaluation-only here.  The [`m252 receipt`](paper/results/raw/m252-tau2-mock-native-v1.json)
hashes the source, ten public mock-base tasks, split metadata, policy, and reset database without
retaining task text or tool outputs.  An oracle replay passes the independent tau2 environment
contract (`1.0`), proving the native verifier is live.

The current 10.52M WebGPU checkpoint was then exposed to the real mock-domain tool schemas for one
agent turn per task.  It emitted `0/10` tool calls, `0/10` exact first actions, and `0/10` bounded
native successes.  This is a useful negative result: the checkpoint can load and the environment
can execute, but the learned LocalAgent catalog/route contract does not transfer to tau2's unseen
customer-service tool names.  It is not a complete tau2 score, retail/telecom result, user-simulator
run, or WebGPU adoption signal.

### τ-Bench selector/grounding ablation (m258)

The follow-up [`m258 ablation`](paper/results/raw/m258-tau2-checkpoint-ablation-v1.json) keeps the
same ten resettable mock tasks and compares three inference arms.  The m46 public Mind2Web child
with its learned selector reaches `0/10` bounded successes; the browser-context child reaches
`2/10`; replacing the closed-world selector with the zero-training schema retriever (`k=1`) reaches
`3/10` on the same source.  The gain is not a benchmark claim: it isolates a deployment failure in
which the learned selector collapses onto `get_users` for unseen tau2 schemas.  A generic schema
grounding fix now extracts structured identifiers such as `task_1`/`user_1` and quoted titles instead
of copying an entire instruction as an argument.

The matched m46→browser tensor audit finds identical configuration/tokenizer and `51` shared
tensors.  Relative movement is largest in the action heads (`71.45%`) and embedding (`24.04%`),
with attention/mixer (`3.20%`), FFN (`4.51%`), and normalization (`0.72%`) moving less.  This supports
reusing the compatible body while replacing or recalibrating the selector for unseen schemas; it
does not prove that the browser child is a better initialization, nor does it justify publishing a
tau2, email, Notion, or WebGPU task-success score.

### τ-Bench bounded airline/retail/telecom probes (m259)

The new [`m259 domain receipts`](paper/results/raw/m259-tau2-airline-m46-retriever-k1-v1.json)
cover eight base tasks each from the public airline (`50` total tasks), retail (`114`), and
telecom (`114`) domains using the same reset-per-task native runner.  A stable gold trajectory
contract passes in all three domains; the telecom contract uses a read-only network-state task
rather than the stateful airplane-mode case whose response sequence is not deterministic under
this isolated constructor.

The m46 checkpoint with the zero-training schema retriever (`k=1`) emits calls on all three
domains, but reaches `0/8` bounded native successes in each.  These are realistic customer-service
and telecom-schema negatives with no external accounts, network services, screenshots, or retained
task text.  They are bounded domain diagnostics, not complete tau2 domain/user-simulator or
leaderboard scores.

The refreshed [`m253 strict gate`](paper/results/raw/m253-workshop-gate-tau2-catalog-refresh-v1.json)
remains `ready: false` with nine missing official native receipts (AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet, ToolSandbox, MCPMark, and
EnterpriseOps-Gym).  AppWorld adapter diagnostics do not substitute for those contracts, and the
existing public model/demo manifest is still the older authenticated artifact rather than this
new child checkpoint.

### Stateful productivity GRPO continuation and closed-loop reality check (m276)

The [`m276 GRPO receipt`](paper/results/raw/m276-stateful-productivity-grpo-v1.json) is a fresh,
bounded pure-PyTorch RL simulation from the current 10.52M BPE browser-context checkpoint. It uses
only the repository's deterministic, source-disjoint local email, Notion, browser-search, recovery,
and abstention state machine; no public benchmark task text, emulator, MCP server, or external
account enters training. A 16-step SFT warm-up plus four GRPO rollout steps raises the held-out
local shaped reward from `0.000` to `0.09375`, but exact action/text accuracy remains `0%`.

The [`m276 runtime receipt`](paper/results/raw/m276-stateful-productivity-runtime-v1.json) keeps
the verifier boundary explicit: the oracle reaches `5/5` complete tasks, while the checkpoint
completes only the abstention task (`1/5`, `20%`). Email, Notion, browser search, and recovery all
remain `0/1`; this is stronger than a teacher-forced metric but is still a local simulation, not
AndroidWorld, BrowserGym, MCPMark, EnterpriseOps-Gym, or real-account evidence.

The paired [`m276 weight audit`](paper/results/raw/m276-stateful-productivity-grpo-weight-v1.json)
confirms identical configuration/tokenizer/shapes and frozen deployment heads. Relative backbone
movement is small but non-zero (embedding `1.89%`, attention/mixer `0.87%`, FFN `1.04%`,
normalization `0.035%`). The adoption decision is unchanged: retain the compatible body only as
an initialization candidate, and do not ship this RL child until service-schema grounding and
native closed-loop success improve against a matched no-transfer control.

### Current ToolACE action-history continuation and free-run control (m277)

The [`m277 transfer receipt`](paper/results/raw/m277-toolace-action-history-transfer-v1.json)
extends the current checkpoint audit to the pinned public [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
action-history projection.  The run uses `256` source-record-disjoint training rows and `64`
held-out rows from revision `6bda777c88d21e5a204703c1ee45597a8fa4f734`, with the tool-response and
non-action prose boundary recorded by the adapter manifest.  Warm continuation preserves compatible
weights and lowers held-out mean loss from `4.655` to `4.250`; random-backbone control remains near
chance (`9.781` to `9.474`).  Warm held-out token accuracy is `47.30%` versus `0.17%` for random,
but both arms remain at `0%` teacher-forced sequence exactness.

The stricter [`warm free-run receipt`](paper/results/raw/m277-toolace-action-history-warm-v1.json)
finds `20%` tool-name exactness, `0%` argument exactness, and `0%` complete action-history episodes
over 16 held-out rows.  The matched [`random free-run receipt`](paper/results/raw/m277-toolace-action-history-random-v1.json)
finds `16.67%` tool-name exactness and `3.33%` step exactness, so the random control wins this tiny
free-run slice despite its much worse teacher-forced loss.  This is a useful deployment warning:
backbone reuse is clearly valuable for representation learning, but it is not sufficient to adopt
the current tool heads or claim reliable multi-turn tool execution.

### Current AppWorld native baseline and public action-step continuation (m278)

The [`m278 baseline receipt`](paper/results/raw/m278-appworld-current-checkpoint-native-v1.json)
reruns the current `10,524,544`-parameter WebGPU checkpoint in the public AppWorld `0.2.0` data
environment.  The runner's own contract test passes, all six caller-selected tasks reset against
isolated task databases, and the native verifier executes; the model nevertheless replays `0/6`
actions and makes `0` API calls.  This is a current-checkpoint zero-action interface baseline, not
an AppWorld leaderboard or email/SMS/Spotify result.

To separate action-program length from routing, the [`m278 continuation report`](paper/results/raw/m278-appworld-action-step-sft-v1.json)
trains only on 24 public AppWorld train rows and evaluates on 12 disjoint public dev rows.  The
source is pinned to AppWorld data `0.2.0` (package `0.2.0.dev0`); credentials, bootstrap calls,
protected test data, and raw solution programs remain outside Git.  Warm continuation raises
held-out first-action token accuracy from `43.28%` to `58.70%` and lowers mean loss from `3.989`
to `2.596`, but sequence exactness remains `0/12`.

The paired [`m278 weight audit`](paper/results/raw/m278-appworld-action-step-weight-v1.json)
finds identical configuration/tokenizer/shapes and frozen action heads, with relative movement of
`0.434%` in the embedding, `0.248%` in attention/mixer, `0.306%` in the FFN, and `0.010%` in
normalization.  The [`m278 native adapter receipt`](paper/results/raw/m278-appworld-action-step-native-v1.json)
then enables strict one-call AST replay and schema grounding on 12 disjoint dev tasks.  The child
still selects no replayable `run_python`/API action: `0/12` native successes, `0` action replays,
and `0` native API calls.  The result is useful source-disjoint transfer evidence, but it does not
close the AppWorld, real-account, or WebGPU closed-loop publication boundary.

### AppWorld route/selector head-only repair and schema failure (m279)

The [`m279 head adapter report`](paper/results/raw/m279-appworld-head-adapter-v1.json) isolates
deployment-head learning from backbone transfer.  It uses the same 24 public AppWorld train rows
and 12 disjoint dev rows as m278, but performs one language-model update and 256 route/selector
updates.  Held-out route and dense-selector top-1 accuracy move from `0%` to `100%` on all 12
rows.  The [`m279 weight audit`](paper/results/raw/m279-appworld-head-weight-v1.json) shows the
intended movement pattern: the shared embedding, attention/mixer, FFN, and normalization groups
remain unchanged, while the serialized action-head group moves `65.93%` relative L2.

The [`m279 native receipt`](paper/results/raw/m279-appworld-head-native-v1.json) forces the
selector-first path and runs strict one-call AST/schema replay on the same 12 disjoint dev tasks.
The adapter now executes `9/12` real AppWorld API calls (`48` total requests including `27`
credential/bootstrap requests), but verifier success remains `0/12`.  The failed calls are a
valuable interface diagnosis: routing can be learned, while API-name and argument/schema grounding
still fail.  This is resettable native execution evidence, not an AppWorld leaderboard score or a
reason to ship the head adapter to WebGPU.

### Longer AppWorld action-code SFT and deployment transfer (m280)

The [`m280 SFT report`](paper/results/raw/m280-appworld-long-sft-v1.json) extends the same public,
source-disjoint AppWorld first-action projection to 256 body updates.  Held-out assistant-token
accuracy reaches `96.64%`, mean loss falls to `0.6915`, and teacher-forced sequence exactness reaches
`9/12` (`75%`) on the 12-row dev projection.  This is a real improvement over m278, but the native
[`run_python` receipt`](paper/results/raw/m280-appworld-long-native-runpython-v1.json) emits no
replayable action on the same prompts, demonstrating a sharp teacher-forcing/free-run gap.

Combining the m280 body with the m279 route/selector heads gives a second native control.  The
[`combined native receipt`](paper/results/raw/m280-appworld-long-heads-native-v1.json) executes all
`12/12` bounded API calls in resettable AppWorld fixtures (`48` requests, including `36` bootstrap
requests), but no complete task verifier passes.  The source-bound [`first-action exactness receipt`](paper/results/raw/m280-appworld-first-action-exactness-v1.json)
compares the translated code hashes with the public dev projection and finds `0/12` exact codes.
The body-only [`weight audit`](paper/results/raw/m280-appworld-long-weight-v1.json) shows relative
movement of `3.579%` in the embedding, `1.405%` in attention/mixer, `1.814%` in the FFN, and
`0.108%` in normalization; the combined [`head audit`](paper/results/raw/m280-appworld-long-heads-weight-v1.json)
adds the expected `65.93%` action-head movement.  The adoption conclusion is unchanged: longer
SFT improves representation and teacher-forced code likelihood, but schema-grounded free-run
execution and multi-step task completion remain open.

### Public-train AppWorld schema retriever control (m281)

The [m281 learned API-head report](paper/results/raw/m281-appworld-api-head-training-v1.json)
trains a five-class frozen-feature app.api head from the same public train rows. It reaches only
6/12 on disjoint dev prompts, so it is retained as a negative control rather than adopted.
The separate [m281 retriever report](paper/results/raw/m281-appworld-api-retriever-v1.json)
uses char-ngram nearest-neighbor examples from the public train projection; it has no learned model
weights and reaches 12/12 API-label accuracy on the same dev rows.

The retriever is wired into the strict evaluator as a schema-candidate restriction, with argument
fields learned only from the train traces (optional query is not hallucinated for unrelated APIs).
The [m281 native receipt](paper/results/raw/m281-appworld-retriever-native-v1.json) replays
12/12 one-step calls (48 requests including 36 fixture bootstrap requests), and the
source-bound [m281 first-action receipt](paper/results/raw/m281-appworld-first-action-exactness-v1.json)
matches all 12/12 translated code hashes against the public dev projection. Full task verifier
success remains 0/12: these tasks require multi-step programs, so this result establishes exact
first-action schema grounding and isolated execution, not complete AppWorld success or WebGPU
productivity readiness.

### Current-checkpoint native BrowserGym canary (m282)

The [m282 BrowserGym canary comparison](paper/results/raw/m282-browsergym-current-checkpoint-canary-v1.json)
runs the current `10.52M`-parameter WebGPU checkpoint in the pinned BrowserGym `0.14.3` /
MiniWoB test environment with Chromium revision `1117`.  Across 20 reset episodes covering five
task families, the accessibility-only arm records `0/20` success, `0.0` reward, and only `10/200`
grounded steps; the model emits `open_app` on 80 steps, `move_cursor` on 40, no parseable tool on
60, and other non-browser actions on the remainder.

The matched DOM-coordinate sidecar is intentionally non-official.  It recovers the deterministic
ascending-numbers family (`4/4`) but leaves bisect-angle, choose-list, click-button, and
click-button-sequence at `0/16`.  This isolates the failure: the checkpoint still needs a
browser-specific action policy and accessibility/target grounding; coordinate geometry alone does
not establish general browser control.  The result is a native canary, not the complete 240-episode
official plan, visual grounding, or an external-account result.

The complete [m282 official-plan receipt](paper/results/raw/m282-browsergym-current-checkpoint-official-v1.json)
then runs all `240` pinned episodes (`60` task families × `4` fixed seeds) with the same checkpoint,
BrowserGym/MiniWoB revisions, Chromium revision, and ten-step limit.  The official-split flag is
verified, but native success is `0/240`, reward is `0.0`, and `2,080/2,400` actions become
`noop(0)`.  Every task family is `0/4`; this is a complete, reproducible negative browser-control
result and closes the current-checkpoint BrowserGym gate against promotion.

### Current export and native WebGPU deployment audit (m283)

The [m283 export audit](paper/results/raw/m283-current-export-deployment-audit-v1.json) rebuilds
the HF-format and ONNX/WebGPU artifacts from the exact `bc1aca…` checkpoint rather than reusing
the stale generated files in the checked-in Space directory.  The HF bundle reloads with exact
PyTorch parity (`max_abs_diff=0`, argmax agreement `1.0`), and the WebGPU export passes the hard
fp32/fp16 graph parity gate.  A clean static Space directory verifies all eight generated files,
manifest hashes, and the `10,524,544`-parameter checkpoint identity.

Publication is still not claimed: `hf auth whoami` reports `Not logged in`, so no Hub model or
Space URL exists.  Two local Chromium probes (headless and headed) also fail closed because the
host exposes no non-empty WebGPU adapter identity; the export is ready, but native WebGPU
throughput and hardware capability remain unverified until run on a suitable GPU host.

### Workshop/publication gate re-audit (m284)

The [m284 gate receipt](paper/results/raw/m284-workshop-gate-current-v1.json) joins the pinned
catalog, current-checkpoint native receipts, export audit, and transfer ablation.  The decision is
`ready=false`, even though the structural checks for catalog coverage, MobileGym, the complete
BrowserGym/MiniWoB plan, the earlier Apple Metal WebGPU receipt, and the parent-head/random
ablation pass.  Nine required native surfaces remain unsupplied: AndroidWorld, MobileSafetyBench,
iOSWorld, OSWorld, OSWorld-V2, AgentNet, ToolSandbox, MCPMark, and EnterpriseOps-Gym.

The public manifest is a real, independently verified older byte-model/Space publication, but it
is not the current BPE checkpoint.  The current export is therefore a local reproducibility
artifact, not a public model release.  This distinction, plus the m283 adapter-identity failure,
keeps the workshop claim closed until the current checkpoint is uploaded and re-fetched
anonymously and the missing native suites are executed with their independent verifiers.

The computer-use source contract is now release-bound as well: OSWorld 2.0 requires the matching
`osworld-v2-2026.06.24` code, task classes, gated assets, and mocked websites.  Without the gated
snapshot and a resettable desktop VM, no OSWorld-V2 result is admitted to the WebGPU or workshop
claims.

### Extended public weight-transfer continuation (m286)

The [m286 receipt](paper/results/raw/m286-cross-surface-public-weight-transfer-v1.json) repeats
the matched source-disjoint AndroidControl/AgentNet experiment for `32` updates over `1,024`
public training rows and `64` held-out rows.  Warm-start token accuracy improves `51.81%`→`57.04%`
(`3.772`→`2.716` loss), with AndroidControl `60.91%`→`66.67%` and AgentNet `45.88%`→`50.76%`.
The random-backbone control reaches `8.71%` after the same budget.  Group movement is
`0.008%`–`0.374%` for the warm backbone versus `7.8%`–`119.7%` for random, so compatible
pretrained weights and a lower backbone learning rate remain the adopted policy.

This is still a teacher-forced text/accessibility result: exact held-out sequences remain `0/64`,
and no native emulator, browser, desktop VM, screenshot, MCP, or external-account side effect was
run.  The child is not promoted to the WebGPU release.

### ToolSandbox runtime-head transfer control (m288)

The [m288 receipt](paper/results/raw/m288-toolsandbox-runtime-head-transfer-v1.json) trains only
the route and dense candidate-selector heads against a pinned public ToolSandbox AST projection:
`107` source-disjoint training rows and `20` held-out rows, with the current `10,524,544`-parameter
body frozen.  The warm parent reaches `80%` selector top-1, `95%` selector top-3, and `100%`
app-action routing on the held-out rows, up from `45%`, `90%`, and `0%`.  The matched random-body
control reaches the same post-update scores, so this bounded projection is evidence of head
capacity and candidate-list regularity, not a claim of useful pretrained-backbone transfer.

Both arms move their heads by roughly `1.27`–`1.49` relative L2 and neither child is promoted.
The projection is AST-only; no ToolSandbox simulator, user simulator, verifier, external API, or
native environment was executed.  Native ToolSandbox evidence remains a publication blocker.

### Public Mind2Web grounding continuation and BrowserGym canary (m289)

The [m289 receipt](paper/results/raw/m289-mind2web-browsergym-transfer-v1.json) continues the
current BPE checkpoint on a public Mind2Web train-only DOM/action projection (`36` train
conversations / `219` decisions and `12` parent/typed-slot-disjoint held-out conversations /
`63` decisions).  Adding browser `target_id`/`value` pointer rows and 64 low-rate updates leaves
held-out exact pointer-span accuracy at `0/63`.

The compatible body moves only `0.286%`–`0.309%` across attention/FFN groups and `0.293%` in the
embedding group.  A separate pinned BrowserGym/MiniWoW accessibility-only canary executes `20`
episodes with `0/20` success.  This is a useful negative result for the WebGPU browser path, not an
official Mind2Web test score, full BrowserGym score, visual grounding result, or account-control
claim; the child is not promoted.

### Mind2Web browser-head and pointer transfer audit (m290–m291)

The [m290–m291 receipt](paper/results/raw/m290-mind2web-browser-head-pointer-v1.json) isolates
browser routing from argument grounding.  Frozen-body warm head transfer raises source-disjoint
Mind2Web selector top-1 `19.05%`→`92.06%` and top-3 `85.71%`→`100%`; the matched random head reaches
`85.71%` top-1.  Warm route/selector movement (`0.674`/`0.820` relative L2) is below random
(`1.253`/`1.133`).

The standard-schema pointer continuation adds `target`/`text` and reaches only `3/63` exact spans.
All three native 20-episode BrowserGym/MiniWoW canaries remain `0/20`; outputs reveal systematic
`type_text` placeholder or prompt-fragment copying.  Pointer precedence now prefers learned spans
when available, but the child is not promoted and native browser capability remains unverified.

### AndroidControl mobile dispatch transfer and native MobileGym canary (m292)

The [m292 receipt](paper/results/raw/m292-mobile-dispatch-transfer-native-v1.json) uses the public
AndroidControl mirror only as a text/action projection: `4,096` train rows and a separate balanced
`904`-row test file, with screenshot bytes omitted.  A frozen-body warm route/selector continuation
reaches `100%` route accuracy and `42.81%` selector top-1; the matched random-head control reaches
`38.94%`.  Warm selection is strong for open-app (`89.6%`) but weak for click (`8.0%`) and zero
for long-press and navigate-home, showing that the public action vocabulary is not equivalent to
screen grounding.

The warm child is evaluated in the pinned MobileGym simulator and independent state-diff judge on
20 official-test tasks.  It completes `0/20` tasks and collapses to `mobile_press_enter` on 19
episodes plus one `mobile_submit_answer`.  The parent’s first 20 tasks in the complete m262 run are
also `0/20` (with a different two-step limit), so this transfer does not improve executable mobile
control.  The result supports retaining parent weights as an initialization candidate while
rejecting this child for WebGPU promotion; it is not an Android emulator, screenshot, or real-device
score.

The [m293 selector-first ablation](paper/results/raw/m293-mobilegym-selector-first-canary-v1.json)
forces the learned top candidate instead of allowing LM re-ranking.  It remains `0/20` with the
same `19` press-enter and one submit-answer collapse.  This is useful diagnosis: MobileGym's
task-state distribution is not recognized by the AndroidControl-trained selector, so a decoding
policy change alone cannot close the mobile grounding gap.  Selector-first remains opt-in and the
child is rejected for deployment.

### Current-checkpoint native ToolSandbox smoke and interactive control (m297)

The [m297 receipt](paper/results/raw/m297-toolsandbox-current-native-smoke-v1.json) is the first
current BPE-checkpoint ToolSandbox native run in this continuation.  The pinned simulator and
milestone verifier execute successfully: a bounded one-step smoke passes `2/3` settings/message
scenarios, with exact `cellular_off` and `wifi_off` milestones but only `0.425` similarity for the
phone-message schema.  A bounded interactive scripted-user run passes `0/3`, showing that the
current tool schema does not survive multi-turn feedback even when native execution is available.

Because the official split and model-based user simulator were not executed, this result is a
native smoke diagnostic rather than an official ToolSandbox score.  It is nevertheless stronger
than the prior text-only projection and keeps the adoption decision negative for the current
checkpoint.

The [m298 workshop gate re-audit](paper/results/raw/m298-workshop-gate-current-toolsandbox-v1.json)
now includes this native receipt and records the exact blocker `official_split_not_verified`; the
overall gate remains closed because the official ToolSandbox split and the other native surfaces are
still absent.

### Current-checkpoint ToolSandbox base-matrix replay (m300)

The [m300 receipt](paper/results/raw/m300-toolsandbox-current-base-v1.json) reruns the same
current 10.52M BPE checkpoint over every one of the 129 pinned base/no-distraction scenarios
(zero simulator or verifier exceptions).  It completes `29/129` (`22.48%`) exact milestones.
The result is sharply category-dependent: insufficient-information cases reach `26/28`, while
multiple-tool-call, multiple-user-turn, canonicalization, and state-dependency cases have no exact
completions under the one-step scripted-user protocol.  This isolates the current failure mode as
stateful continuation and argument grounding rather than package availability.

This is the complete upstream base set, not the 1,032 runtime variants: distraction and schema
scramble variants remain unevaluated, and the model-based user simulator is still absent.  Therefore
m300 strengthens native diagnosis but cannot satisfy the official ToolSandbox/publication gate.

The [m301 gate](paper/results/raw/m301-workshop-gate-current-toolsandbox-base-v1.json) joins m300
without relaxing the contract: ToolSandbox is still blocked specifically by
`official_split_not_verified`, while the catalog, current MobileGym, BrowserGym/MiniWoB, WebGPU,
weight-ablation, and existing public-artifact checks remain independently visible.

### Current-checkpoint binding enforcement (m317)

The [m317 gate](paper/results/raw/m317-workshop-gate-current-checkpoint-bound-v1.json) closes a
lineage gap in the final join: when a current checkpoint is supplied, every native receipt must
carry the same SHA-256 directly or under its checkpoint object.  The current MobileGym receipt
(`1/256`) and BrowserGym/MiniWoB receipt (`0/240`) therefore pass only after binding to
`bc1aca…`; older receipts without a checkpoint identity are no longer eligible by accident.  The
gate remains `ready=false` with ten blockers: nine required native families are still absent and
the public Hub manifest is a legacy byte-model release without the current checkpoint binding.

### Source-corrected current gate (m318)

The [m318 gate](paper/results/raw/m318-workshop-gate-source-corrected-v1.json) repeats the same
join after correcting two catalog citations discovered during an official-source audit:
AndroidWorld now points to its actual `2405.14573` paper, and BrowserGym points to the official
2025 ecosystem paper (`openreview.net/forum?id=5298fKGmv3`) rather than unrelated arXiv records.
The catalog fingerprint is `b1b170b2…`; all native and public-release blockers remain explicit, and
the gate is still `ready=false`.

### Current-checkpoint RL preflight (m319)

The [m319 receipt](paper/results/raw/m319-current-rl-preflight-v1.json) tests the actual
`sft_browser_context` checkpoint rather than the older pilot parent.  The lineage rule now
accepts this specialized SFT stage because its checkpoint lineage explicitly remains `stage: sft`.
The isolated two-step RL prefix still fails closed: every sampled rollout receives zero reward,
there is no reward diversity, no nonzero-LR policy update changes a model tensor, and held-out
tool exactness remains `0%`.  This is a useful training diagnosis—not a promotion: the current
checkpoint needs a schema-valid, learnable reward curriculum before RL can be claimed or exported.

### Current-checkpoint stateful RL preflight (m321)

The [m321 receipt](paper/results/raw/m321-current-stateful-rl-preflight-v1.json) supplies the
learnable curriculum without weakening the canonical failure: it uses only the repository’s
resettable email/Notion/browser fixture and shaped schema/tool/argument/transition rewards.  The
same current checkpoint and tokenizer produce two reward values, change 40 policy tensors, and
realize two optimizer updates in the isolated prefix.  No benchmark text, emulator, account, or
external side effect is used, so this is a valid RL-simulation pass—not a native quality claim.

### RL-bound workshop gates (m320–m322)

The [m320 gate](paper/results/raw/m320-workshop-gate-rl-bound-v1.json) made that diagnosis part
of the publication contract.  The successful [m321 receipt](paper/results/raw/m321-current-stateful-rl-preflight-v1.json)
then satisfied the RL requirement without changing the canonical m319 result.  The [m322 gate](paper/results/raw/m322-workshop-gate-stateful-rl-v1.json)
therefore has ten blockers rather than eleven: RL now passes, while nine native benchmark
families and the current-checkpoint public artifact remain unresolved.

### Longer stateful continuation and weight audit (m323)

The [m323 control](paper/results/raw/m323-current-stateful-long-continuation-v1.json) extends the
same current parent with 64 stateful SFT updates and 8 shaped-reward RL steps.  It produces a
healthy six-value reward distribution and raises mean reward slightly (`0.3313→0.3500`), but
held-out exactness falls from `25%` to `12.5%`; the deployment-shaped runtime remains `0/5`
completed tasks.  The paired [weight report](paper/results/raw/m323-current-stateful-long-weight-v1.json)
finds compatible configuration/tokenizer/shapes, zero action-head movement, and much larger body
movement—embedding `7.59%`, attention/mixer `1.97%`, FFN `2.52%`.  This is a negative control,
not a promotion: longer local RL is not evidence of better realistic-agent behavior, and the
adoption recipe remains a verified body with low-rate updates plus separately controlled heads.

### Current-checkpoint low-rate transfer ablation (m326)

The [m326 receipt](paper/results/raw/m326-current-stateful-lowrate-transfer-v1.json) reruns the
current parent with matched frozen, low-rate-unfrozen, and random-backbone arms on the same
source-disjoint local email/Notion/browser/recovery fixture.  The low-rate arm keeps backbone
movement below `0.2%` and raises mean shaped reward to `0.2344` and exact tool selection to `37.5%`,
but closed-loop completion is unchanged at `1/5` tasks (`0/4` productive workflows); the frozen
arm reaches `25%` exact tools and the random arm `12.5%`.  Selector top-1 is not improved over the
frozen arm, so the receipt explicitly rejects capability adoption.  The result strengthens the
weight policy—small body updates are safer than m323's larger continuation—but still does not
establish native mobile, browser, desktop, MCP, email, Notion, or WebGPU task success.

## Publication checklist

- [ ] Official source revision, license, split, task IDs, and byte/hash receipt.
- [ ] Train/eval parent and typed-slot disjointness; benchmark canaries absent from training.
- [ ] Native environment executed with complete action/termination logs and independent verifier.
- [ ] WebGPU hardware adapter, TTFA/throughput, memory, export parity, and closed-loop action
      success measured separately from token accuracy.
- [ ] Public model/demo manifest binds the exact current checkpoint SHA-256; an older public
      artifact must not satisfy the current-release gate.
- [ ] Warm-start, low-rate, and matched-random transfer arms with per-group weight movement.
- [ ] No real email, Notion, GitHub, or browser side effect without an isolated resettable fixture.

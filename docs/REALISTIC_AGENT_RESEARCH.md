# Realistic agent evaluation research memo

Status: protocol refresh on 2026-08-03. This memo records the public benchmark methods that
matter for a sub-100M text-first WebGPU agent. It is a source and protocol guide, not a claim that
the repository has completed every benchmark.

## What the public benchmarks actually measure

| Surface | Public source and method | What LocalAgent may claim locally |
|---|---|---|
| Mobile UI | [AndroidWorld](https://github.com/google-research/android_world) runs resettable Android emulator tasks with accessibility/screenshot observations and durable task rewards. | Accessibility-tree/text protocol tests are useful for routing; an Android emulator, ADB, official task set, and reward logs are required for a native score. |
| Personalized mobile UI | [iOSWorld](https://iosworld.io/) provides 133 tasks over 26 interconnected iOS apps with persistent seeded user identity and an optional MCP server. | Treat identity, cross-app state, and MCP-vs-GUI as separate axes; a WebGPU text projection cannot claim native iOS control or personalization. |
| Mobile safety | [MobileSafetyBench](https://mobilesafetybench.github.io/) evaluates Android-device safety, harmful side effects, and indirect prompt injection in messaging/banking-style tasks. | Run a dedicated refusal/confirmation/safe-side-effect gate before enabling email, messaging, settings, or payment tools. |
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
current research inventory against the 40-row canonical catalog and 21-row supplemental registry.
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

## Publication checklist

- [ ] Official source revision, license, split, task IDs, and byte/hash receipt.
- [ ] Train/eval parent and typed-slot disjointness; benchmark canaries absent from training.
- [ ] Native environment executed with complete action/termination logs and independent verifier.
- [ ] WebGPU hardware adapter, TTFA/throughput, memory, export parity, and closed-loop action
      success measured separately from token accuracy.
- [ ] Warm-start, low-rate, and matched-random transfer arms with per-group weight movement.
- [ ] No real email, Notion, GitHub, or browser side effect without an isolated resettable fixture.

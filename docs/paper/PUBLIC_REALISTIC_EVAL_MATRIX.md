# Public realistic-agent evaluation matrix

Status: source discovery and split policy refreshed on 2026-08-03. This is the reference map for
the WebGPU agent; it is not a claim that every suite has been executed. The machine-readable
source of truth is [`realistic-agent-public-eval-matrix.v1.json`](../../configs/data/realistic-agent-public-eval-matrix.v1.json).
The source-method and native-vs-projection protocol is summarized in the
[`realistic-agent research memo`](../REALISTIC_AGENT_RESEARCH.md).

## How to read the matrix

The rows intentionally separate three evidence types:

1. **Trainable public data** has an explicit public download, reviewed license, official train
   partition, and a source-specific adapter. Current candidates are AndroidControl, Android in the
   Wild, Mind2Web train, and AgentNet train. A downloaded artifact still needs a byte/hash receipt.
2. **Runtime evaluation** is not SFT data. AndroidWorld, MobileGym, MobileWorld, BrowserGym,
   OSWorld, MCPMark, ToolSandbox, and AppWorld require their own resettable environment and
   verifier. A metadata inventory or local state-machine proxy is never an official score.
3. **Protected/terms-review sources** are useful for protocol design but cannot be copied into a
   public model bundle until their data and service terms are reviewed. This includes WebLINX,
   Computer Agent Arena, Mobile-Bench, τ-Bench, and several visual/desktop suites.

The WebGPU projection column is a deployment recommendation, not a modality equivalence claim:
an accessibility-tree or DOM text projection can test routing and state transitions, but it cannot
prove screenshot grounding. Native runs must publish the observation contract, tool catalog, model
hash, tokenizer hash, runtime versions, task IDs, seeds, and complete action/termination logs.

## Source-linked inventory

| Family | Source | What it measures | Train policy | WebGPU projection | Local status |
|---|---|---|---|---|---|
| Mobile | [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) | Human mobile demonstrations, accessibility/gesture actions | Train official train split only | Goal + compact accessibility tree | Adapter ready; bounded replay only |
| Mobile | [Android in the Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild) | Large visual/gesture mobile episodes | Train official task/app/platform train split only | Instruction + text screen projection | Adapter ready; no emulator score |
| Mobile | [AndroidWorld](https://github.com/google-research/android_world) | 116 tasks, 20 apps, dynamic emulator rewards | Eval-only runtime | Accessibility tree + goal | Metadata inventory; emulator pending |
| Mobile | [MobileGym](https://github.com/Purewhiter/mobilegym) | 28 simulated apps, 416 parameterized tasks, deterministic judges | Eval-only until release/license pin | Structured JSON state + goal | Not started |
| Mobile | [MobileSafetyBench](https://github.com/jylee425/mobilesafetybench) | 100 Android-emulator tasks spanning helpfulness, safety, and indirect prompt injection | Eval-only; safety prompts, emulator snapshots, APKs, and attack variants stay out of SFT | Goal + compact image/text state + refusal/confirmation policy | Source pinned; native emulator pending |
| Mobile | [iOSWorld](https://github.com/ljang0/iOSWorld) | 133 tasks across 26 seeded iOS apps, including multi-app and memory/personalization workflows | Eval-only; seeded identity, apps, task files, and rubrics stay out of SFT | Compact identity/state + MCP schema; visual phone control requires an encoder | Source pinned; iOS Simulator pending |
| Mobile | [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) | 201 long-horizon agent-user/MCP tasks across 20 apps | Eval-only until release split pin | State text + explicit tools | Not started |
| Mobile | [MobileAgentBench](https://mobileagentbench.github.io/) | 100 tasks across 10 open-source apps | Eval-only | Goal + accessibility/text projection | Not started |
| Mobile | [Mobile-Bench](https://github.com/XiaoMi/MobileBench) | Device UI and API command evaluation | Eval-only; terms review | Structured UI/API text | Not started |
| Browser | [BrowserGym / MiniWoB++](https://github.com/ServiceNow/BrowserGym) | Closed-loop DOM/A11y browser episodes | Runtime-only | Goal + compact DOM/A11y | Native 240-episode receipts |
| Browser | [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | Grounded CLICK/TYPE/SELECT web trajectories | Train official train only; test protected | Cleaned DOM + candidate grounding | Public-train continuations measured |
| Browser | [WebLINX](https://huggingface.co/datasets/McGill-NLP/WebLINX) | Multi-turn browser dialogue and element grounding | Restricted; legal/privacy review | Text/action-history projection | Metadata/privacy audit |
| Browser | [WebArena](https://github.com/web-arena-x/webarena) / [VisualWebArena](https://github.com/web-arena-x/visualwebarena) | Multi-site web state and visual navigation | Eval-only runtime | DOM-only diagnostic; visual score pending | Not started |
| Browser | [WebBench](https://github.com/Halluminate/WebBench) | 2,454 realistic READ/CREATE/UPDATE/DELETE/file workflows across 452 live websites | Eval-only; live tasks and credentials stay outside training | DOM/accessibility action protocol only | Cataloged; release/runtime pin pending |
| Browser | [BU Bench V1](https://github.com/browser-use/benchmark) | 100 encrypted browser tasks spanning WebBench, Mind2Web 2, BrowseComp, GAIA, and custom tasks | Eval-only; encrypted task text never enters SFT | Tool routing/DOM safety canary only | Cataloged; task decryption/runtime intentionally not run |
| Computer | [AgentNet / OpenCUA](https://huggingface.co/datasets/xlangai/AgentNet) | Cross-OS screenshot/action trajectories | Train official train parents only | Text action vocabulary; no visual claim | Bounded continuation and ablation measured |
| Computer | [Computer Agent Arena](https://github.com/xlang-ai/computer-agent-arena) | Human preference and long-horizon desktop trajectories | Eval-only; deduplication required | Instruction/action-family diagnostic | Metadata + text-only probe |
| Computer | [OSWorld](https://github.com/xlang-ai/OSWorld) | Real desktop VM task execution | Runtime-only | Protocol replay only | VM pending |
| Computer | [OSWorld 2.0](https://github.com/xlang-ai/OSWorld-V2) | Release-pinned long-horizon desktop tasks | Eval-only; gated assets | Protocol replay only | Not started |
| Tool API | [Apple ToolSandbox](https://github.com/apple/ToolSandbox) | Stateful, conversational tool dependencies and milestones | Eval-only; source terms review | State/canonicalization contract | AST metadata profile |
| Tool API | [MCPMark](https://github.com/eval-sys/mcpmark) | Notion, GitHub, filesystem, Postgres, Playwright MCP workflows | Eval-only; isolated services | Routing/schema/state proxy | Metadata + service-contract probe |
| Tool API | [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) | 512-tool enterprise state and SQL verification | Eval-only; server/verifier required | Schema retrieval + local state proxy | Card/API metadata profile |
| Tool API | [AppWorld](https://appworld.dev/) | 9 simulated apps, 457 APIs, 750 executable tasks | Eval-only | Compact API/state projection | Not started |
| Tool API | [τ-Bench](https://github.com/sierra-research/tau2-bench) | Policy-following user-agent-tool conversations | Eval-only | Text multi-turn tool loop | Not started |
| Terminal | [TUA-Bench](https://tuabench.ai/) | Deterministic general-purpose terminal tasks | Eval-only | Command/tool schema projection | Not started |
| Tool API | [BFCL](https://github.com/ShishirPatil/gorilla) | Stateless schema, argument, parallel-call, and abstention control | Eval-only checker; prompt denylist for corpus | Compact function catalog | Provenance/audit adapters |

## What is actually trainable now

The current data path is intentionally narrow and reproducible:

```text
public train split -> source hash + split receipt -> source adapter
    -> Conversation -> BPE SFT / continuation -> frozen held-out probe
```

The current public continuations use bounded Mind2Web train and AgentNet train records. They show
teacher-forced or selector improvements, but not native benchmark success. The stateful email,
Notion, browser-search, 404-recovery, and abstention probe in
[`stateful_productivity.py`](../../src/localagent/data/stateful_productivity.py) is deliberately
synthetic and local; it tests closed-loop mechanics without copying public benchmark task text or
touching real accounts.

## Acceptance gates before publishing a capability claim

- **Data:** source revision, license evidence, official split, byte/hash receipt, privacy filter,
  and task-disjoint train/eval IDs.
- **Model transfer:** identical config/tokenizer hashes; compare pretrained frozen, pretrained
  low-rate-unfrozen, and matched-random arms; report per-family tensor movement and held-out
  state/action metrics.
- **Runtime:** native environment, exact observation/action contract, reset seed, complete logs,
  and independent run-level uncertainty.
- **WebGPU:** tokenizer/marker/padding contract, export parity, TTFA p50/p95, memory, and
  complete-action success—not decoded token rate alone.
- **Claims:** never turn a metadata profile, replay, text-only projection, or local state machine
  into an official AndroidWorld, OSWorld, BrowserGym, MCPMark, ToolSandbox, or leaderboard score.

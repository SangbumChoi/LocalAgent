# Realistic agent evaluation research memo

Status: protocol refresh on 2026-08-02. This memo records the public benchmark methods that
matter for a sub-100M text-first WebGPU agent. It is a source and protocol guide, not a claim that
the repository has completed every benchmark.

## What the public benchmarks actually measure

| Surface | Public source and method | What LocalAgent may claim locally |
|---|---|---|
| Mobile UI | [AndroidWorld](https://github.com/google-research/android_world) runs resettable Android emulator tasks with accessibility/screenshot observations and durable task rewards. | Accessibility-tree/text protocol tests are useful for routing; an Android emulator, ADB, official task set, and reward logs are required for a native score. |
| Personalized mobile UI | [iOSWorld](https://iosworld.io/) provides 133 tasks over 26 interconnected iOS apps with persistent seeded user identity and an optional MCP server. | Treat identity, cross-app state, and MCP-vs-GUI as separate axes; a WebGPU text projection cannot claim native iOS control or personalization. |
| Mobile safety | [MobileSafetyBench](https://mobilesafetybench.github.io/) evaluates Android-device safety, harmful side effects, and indirect prompt injection in messaging/banking-style tasks. | Run a dedicated refusal/confirmation/safe-side-effect gate before enabling email, messaging, settings, or payment tools. |
| Curated mobile control | [AndroidControl-Curated](https://github.com/batechworks/AndroidControl_Curated) purifies AndroidControl task ambiguity and reports a matched curated split. | Compare original and curated tasks with identical identities/seeds; do not mix the curated benchmark's evaluation rows into SFT. |
| Browser | [BrowserGym](https://github.com/ServiceNow/BrowserGym) exposes MiniWoB, WebArena, WorkArena, VisualWebArena, WebLINX, and related environments through Gymnasium/Playwright. | DOM/accessibility grounding can be tested in the existing MiniWoB runner; screenshot or live multi-site claims require the matching runtime and task release. |
| Real web trajectories | [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) contains more than 2,000 tasks across 137 websites and 31 domains, with public training data and protected test splits. | Train only on the public train partition; keep test tasks and canary strings out of all corpora. Report DOM/action replay separately from native live-web success. |
| Realistic live-site browser workflows | [WebBench](https://github.com/Halluminate/WebBench) covers 2,454 READ/CREATE/UPDATE/DELETE/file tasks across 452 live websites, including authentication, forms, and downloads. | Keep the live task set and credentials evaluation-only; use DOM/action safety canaries or a vendor-approved resettable environment rather than training on the tasks. |
| Contamination-resistant browser canary | [BU Bench V1](https://github.com/browser-use/benchmark) contains 100 encrypted tasks drawn from WebBench, Mind2Web 2, BrowseComp, GAIA, and custom challenges. | Preserve encryption and task text out of SFT and published artifacts; only run a release-matched browser evaluation with decrypted data held locally. |
| Desktop computer use | [OSWorld](https://github.com/xlang-ai/OSWorld) uses real desktop VMs and execution-based evaluators; [OSWorld 2.0](https://osworld-v2.xlang.ai/) adds 108 long-horizon workflows with dynamic environments, cross-source reasoning, implicit state, and visual precision. | A text/accessibility projection is a diagnostic only. A publication score needs the release-matched VM, assets, initial state, action log, and evaluator. |
| Screenshot/action trajectories | [OpenCUA AgentNet](https://github.com/xlang-ai/OpenCUA) provides cross-OS computer-use trajectories; AgentNetBench is an offline representative task suite. | The m47–m62 projection measures action priors and text routing only. Visual grounding requires the embedded images and the upstream AgentNetBench evaluator. |
| GUI plus MCP control | [OSWorld-MCP](https://github.com/X-PLUG/OSWorld-MCP) measures GUI actions, MCP invocation, and decisions together across 158 validated tools and seven desktop applications. | Report tool-invocation rate separately from GUI completion; a schema-routing probe is not an OSWorld-MCP score. |
| Stateful local tools | [Apple ToolSandbox](https://github.com/apple/ToolSandbox) evaluates stateful, conversational tool execution with a user simulator and milestone DAGs, including state dependency, canonicalization, and insufficient-information cases. | Static AST rows can train schemas and measure constrained dispatch; only the simulator plus milestone verifier can establish ToolSandbox success. |
| Stateful MCP services | [MCPMark](https://github.com/eval-sys/mcpmark) runs isolated Notion, GitHub, filesystem, Postgres, and Playwright services with strict verification. Its current repository identifies MCPMark Verified as the default task set. | Tool/schema retrieval and local state contracts are preflight evidence. Native claims require the matching verified release, isolated services, verifier output, and pass@k aggregation. |
| Enterprise email/tools | [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) exposes large tool catalogs and stateful SQL-verifier workflows across enterprise domains. | Name/schema retrieval is a useful failure profile; task success requires the containerized MCP servers and SQL verifiers. |

The recurring methodological point is that a benchmark is not just a prompt list. The authoritative
score includes the observation contract, reset state, action interface, environment revision, and
verifier. A static conversation projection must be labelled as such even when it uses the original
task text.

The new mobile and MCP suites reinforce the same design requirement. iOSWorld's persistent identity
and cross-app data make memory/state tracking first-class rather than an optional prompt feature;
MobileSafetyBench makes confirmation, refusal, and prompt-injection handling measurable; and
OSWorld-MCP separates the decision to invoke a structured tool from the GUI action itself. These
should be represented as distinct WebGPU heads/metrics, not collapsed into token accuracy.

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

## Publication checklist

- [ ] Official source revision, license, split, task IDs, and byte/hash receipt.
- [ ] Train/eval parent and typed-slot disjointness; benchmark canaries absent from training.
- [ ] Native environment executed with complete action/termination logs and independent verifier.
- [ ] WebGPU hardware adapter, TTFA/throughput, memory, export parity, and closed-loop action
      success measured separately from token accuracy.
- [ ] Warm-start, low-rate, and matched-random transfer arms with per-group weight movement.
- [ ] No real email, Notion, GitHub, or browser side effect without an isolated resettable fixture.

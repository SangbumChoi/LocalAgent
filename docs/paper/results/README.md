# Paper result artifacts

This index contains deterministic **untrained-random-weight latency artifacts**, bounded
**trained pretraining-quality proxies**, and one seed-2027
**midtrain → SFT → offline-RL → WebGPU action pilot**. The pilot is a negative capability result:
the fixed-512-token structured policy is fast but collapses to abstention. It must not be read as
useful browser-agent performance.

Separately, the private 1,880-byte four-suite prompt-only decontamination manifest now binds BFCL,
BrowserGym, Mind2Web, and WebLINX outputs totaling 34,068,955 bytes. Its file SHA-256 is
`3466e9242ccc3aadf487fd2c7fa1dc7bdc9ed14a37007955f75cfece0c040ad1` and its canonical self-hash
is `826f53f5699f3c4b8f311a9fe70561f5b7d9aa99ce42ce250548e46aa644010b`. This is not a
benchmark result; protected prompt outputs remain private and ignored.

The final paper-all corpus is also independently freeze-verified: 504,010 retained documents,
523,358,082 training tokens, 5,311,528 validation tokens, and 528,669,610 total tokens. Its bounded
audit screened the explicitly supplied 19,334 normalized denylist prompts with 8,633,077 candidate
checks and removed 15 documents. The audit is explicitly non-exhaustive and is neither a native
benchmark evaluation nor a training result.

## Artifact index

| Measurement | Matched configs | Summary | Byte-for-byte raw browser payloads |
|---|---|---|---|
| Bounded public Mind2Web train normalization + BPE continuation | [`realistic-agent catalog`](../../../configs/data/realistic-agent-eval.catalog.yaml), [`training plan`](../../../configs/data/realistic-agent-training.example.yaml) | [`mind2web-public-train-sample-v1.json`](raw/mind2web-public-train-sample-v1.json) | No raw source bytes; receipt only |
| Mind2Web-adapted 10.5M WebGPU structured bundle | Same BPE parent plus bounded public continuation | [`m11-webgpu-mind2web-bpe-child.json`](raw/m11-webgpu-mind2web-bpe-child.json) | In-app-browser receipt; synthetic state loop only |
| AndroidControl/AITW-adapted 10.5M no-guard dense dispatch | Same BPE parent plus bounded mobile continuation and productivity rows | [`m12-webgpu-mobile-productivity-v10.json`](raw/m12-webgpu-mobile-productivity-v10.json) | In-app-browser receipt; synthetic state loop only |
| Mobile dispatch pretrained-head transfer ablation | Same bounded AndroidControl/AITW rows, parent-head vs seeded-random probes | [`m13-mobile-dispatch-transfer-ablation.json`](raw/m13-mobile-dispatch-transfer-ablation.json) | Offline selector metrics; not a runtime score |
| EnterpriseOps-Gym public email retrieval diagnostic | Frozen v10 dense selector; 67 matched oracle/plus-15 email rows at pinned HF revision | [`m14-enterpriseopsgym-email-retrieval-v10.json`](raw/m14-enterpriseopsgym-email-retrieval-v10.json) | Name-only retrieval; verifiers/server config dropped; no tool execution |
| Stateful mobile/productivity trajectory gate | v11 parent-head continuation; three local workflows and 13 ordered state transitions in an explicit WebGPU browser session | [`m15-webgpu-mobile-productivity-trajectories-v11.json`](raw/m15-webgpu-mobile-productivity-trajectories-v11.json) | 13/13 schema-valid but 0/13 exact and 0/3 complete trajectories; synthetic only |
| Corrected stateful mobile/productivity trajectory gate | v14 probe with 23 state-conditioned rows, state-action grounding fixes, and projected `{tool,args}` scoring | [`m16-webgpu-mobile-productivity-trajectories-v14.json`](raw/m16-webgpu-mobile-productivity-trajectories-v14.json) | Guarded dense 4/13 exact and 3/13 closed-loop; no-guard dense 1/13; no-guard retrieval 2/13; pass@1 remains 0/3 |
| AgentNet offline adapter sample | Official OpenCUA AgentNetBench sample; literal action parser and provenance-bound eval ingestion | [`m17-agentnet-offline-adapter-sample-v1.json`](raw/m17-agentnet-offline-adapter-sample-v1.json) | 1 record, 10 text observations, 12 low-level actions normalized; adapter evidence only, no AgentNetBench score |
| AgentNet scoring proxy replay | Same pinned official sample replayed through the dependency-free AgentNetBench-compatible scorer | [`m18-agentnet-scoring-proxy-sample-v1.json`](raw/m18-agentnet-scoring-proxy-sample-v1.json) | Ground-truth replay total 1.0; scorer sanity check only, not model accuracy |
| AgentNet metadata profile | Pinned Hugging Face metadata inventory for split/domain/device coverage | [`m19-agentnet-metadata-profile-v1.json`](raw/m19-agentnet-metadata-profile-v1.json) | 22,532 unique rows; no trajectory/image payload consumed; inventory only |
| Toolathlon-GYM configuration profile | Pinned public finalpool task-config inventory for realistic MCP/email/Notion/browser coverage | [`m20-toolathlon-gym-config-profile-v1.json`](raw/m20-toolathlon-gym-config-profile-v1.json) | 503 tasks, 25 MCP servers; prompts/workspaces/verifiers omitted; inventory only |
| MCPMark metadata profile | Pinned standard/easy task metadata for Notion, browser, filesystem, GitHub, and database MCP evaluation | [`m21-mcpmark-metadata-profile-v1.json`](raw/m21-mcpmark-metadata-profile-v1.json) | 239 rows (169 standard, 70 easy); descriptions/state assets/verifiers omitted; inventory only |
| MCPMark result aggregation bridge | Strict parser for complete local MCPMark result directories | [`mcpmark.py`](../../../src/localagent/eval/mcpmark.py) | Computes pass@1/pass@k/pass^k only when all expected task/run records exist; no local score yet |
| ToolSandbox result aggregation bridge | Strict parser for upstream stateful milestone result summaries | [`toolsandbox.py`](../../../src/localagent/eval/toolsandbox.py), [`aggregate_toolsandbox.py`](../../../scripts/aggregate_toolsandbox.py) | Hashes and validates scenario records/category coverage; no live ToolSandbox score yet |
| BrowserGym episode aggregation bridge | Strict parser for normalized Gymnasium task/seed episode logs | [`browsergym.py`](../../../src/localagent/eval/browsergym.py), [`aggregate_browsergym.py`](../../../scripts/aggregate_browsergym.py) | Reward/success/action-error diagnostics with exact case coverage; no live browser score yet |
| tau2-bench trajectory/result aggregation bridge | Strict parser for upstream monolithic or directory-based Results JSON | [`tau2.py`](../../../src/localagent/eval/tau2.py), [`aggregate_tau2.py`](../../../scripts/aggregate_tau2.py) | Upstream reward≈1 success and combinatorial Pass^k per task with exact task/trial coverage; no live tau2 score yet |
| AndroidControl/AITW normalized mobile action bridge | Dependency-free offline scorer for normalized `mobile_*` predictions | [`mobile.py`](../../../src/localagent/eval/mobile.py) | Tool/exact-action/trajectory/coordinate diagnostics only; no emulator reward or official device score |
| AndroidControl/AITW mobile scorer replay | Ground-truth replay over bounded normalized public training slices | [`m22-mobile-action-score-replay-v1.json`](raw/m22-mobile-action-score-replay-v1.json) | 26 records / 130 calls; exact replay sanity check only, not model accuracy |
| AndroidWorld emulator-result bridge | Strict parser for upstream gzip-pickle `run_...` checkpoint directories | [`androidworld.py`](../../../src/localagent/eval/androidworld.py), [`aggregate_androidworld.py`](../../../scripts/aggregate_androidworld.py) | Requires expected task coverage and, for full upstream episodes, explicit trusted-pickle acknowledgement; no device score yet |
| BrowserGym/MiniWoB reset-goal reproducibility | [`paper benchmark plan`](../../../configs/data/evaluation-benchmarks-paper.yaml), [`capture runtime`](../../../configs/data/browsergym-capture-runtime-darwin-arm64-py312.json) | [`browsergym-miniwob-reset-capture-20260728.reproducibility.json`](browsergym-miniwob-reset-capture-20260728.reproducibility.json) | Private ignored capture/receipt files; tracked report binds their exact identities |
| Mind2Web protected archive, bounded v2 export, and v3 freeze | [`paper benchmark plan`](../../../configs/data/evaluation-benchmarks-paper.yaml) | [`mind2web-protected-archive-20260728.reproducibility.json`](mind2web-protected-archive-20260728.reproducibility.json) | Private ignored archive, members, and prompt output; tracked report publishes only aggregate/hash evidence |
| 34M hidden-only backbone | [`webgpu-35m-hybrid`](../../../configs/model/webgpu-35m-hybrid.yaml), [`webgpu-35m-attn`](../../../configs/model/webgpu-35m-attn.yaml) | [`m5-webgpu-backbone-20260728.summary.json`](m5-webgpu-backbone-20260728.summary.json) | [`run1`](raw/m5-webgpu-backbone-run1.json), [`run2`](raw/m5-webgpu-backbone-run2.json), [`run3`](raw/m5-webgpu-backbone-run3.json) |
| 34.2M cache-bearing decode | [`webgpu-35m-hybrid`](../../../configs/model/webgpu-35m-hybrid.yaml), [`webgpu-35m-attn`](../../../configs/model/webgpu-35m-attn.yaml) | [`m5-webgpu-cached-decode-20260728.summary.json`](m5-webgpu-cached-decode-20260728.summary.json) | [`run1`](raw/m5-webgpu-cached-decode-20260728-run1.json), [`run2`](raw/m5-webgpu-cached-decode-20260728-run2.json), [`run3`](raw/m5-webgpu-cached-decode-20260728-run3.json) |
| 15.6M cache-bearing decode | [`webgpu-16m-hybrid`](../../../configs/model/webgpu-16m-hybrid.yaml), [`webgpu-16m-attn`](../../../configs/model/webgpu-16m-attn.yaml) | [`m5-webgpu-cached-decode-16m-20260728.summary.json`](m5-webgpu-cached-decode-16m-20260728.summary.json) | [`run1`](raw/m5-webgpu-cached-decode-16m-20260728-run1.json), [`run2`](raw/m5-webgpu-cached-decode-16m-20260728-run2.json), [`run3`](raw/m5-webgpu-cached-decode-16m-20260728-run3.json) |
| 10.5M cache-bearing decode | [`webgpu-10m-hybrid`](../../../configs/model/webgpu-10m-hybrid.yaml), [`webgpu-10m-attn`](../../../configs/model/webgpu-10m-attn.yaml) | [`m5-webgpu-cached-decode-10m-20260728.summary.json`](m5-webgpu-cached-decode-10m-20260728.summary.json) | [`run1`](raw/m5-webgpu-cached-decode-10m-20260728-run1.json), [`run2`](raw/m5-webgpu-cached-decode-10m-20260728-run2.json), [`run3`](raw/m5-webgpu-cached-decode-10m-20260728-run3.json) |
| 10.5M one-TPP pretraining, seed 2026 | [`hybrid model`](../../../configs/model/webgpu-10m-hybrid.yaml) / [`train`](../../../configs/train/pretrain-webgpu-proxy-1tpp-hybrid-seed2026.yaml); [`attention model`](../../../configs/model/webgpu-10m-attn.yaml) / [`train`](../../../configs/train/pretrain-webgpu-proxy-1tpp-attn-seed2026.yaml) | [`webgpu-proxy-1tpp-10m-seed2026.summary.json`](webgpu-proxy-1tpp-10m-seed2026.summary.json) | Not a browser run |
| 10.5M one-TPP pretraining, seeds 2027–2029 | Matched hybrid/attention seed configs | [`webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json`](webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json) | [`scorecards`](raw/pretrain-proxy-seeds2027-2029/) |
| 10.5M trained pretrain-only cache-bearing decode | Same exact checkpoints as the one-TPP row | [`m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json`](m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json) | [`run1`](raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run1.json), [`run2`](raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run2.json), [`run3`](raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run3.json) |
| Seed-2027 stage-pilot internal protocol | Provisional 10.5M hybrid | [`webgpu-proxy-pilot-seed2027.protocol.json`](webgpu-proxy-pilot-seed2027.protocol.json) | Local prespecification only; no independent pre-outcome timestamp is claimed |
| Seed-2027 midtrain/SFT/offline-RL pilot | Provisional 10.5M hybrid | [`webgpu-proxy-pilot-seed2027.summary.json`](webgpu-proxy-pilot-seed2027.summary.json) | Not a browser run |
| Seed-2027 SFT structured action, pre-assistant-padding stress at 512 tokens | Exact SFT checkpoint/action graph bound in summary | [`m5-webgpu-sft-action-pilot-seed2027.summary.json`](m5-webgpu-sft-action-pilot-seed2027.summary.json) | [`run1`](raw/m5-webgpu-sft-action-pilot-seed2027-run1.json), [`run2`](raw/m5-webgpu-sft-action-pilot-seed2027-run2.json), [`run3`](raw/m5-webgpu-sft-action-pilot-seed2027-run3.json) |
| Seed-2027 SFT local DOM loop, same 512-token stress | Same SFT checkpoint/action graph | [`m5-webgpu-sft-dom-pilot-seed2027.summary.json`](m5-webgpu-sft-dom-pilot-seed2027.summary.json) | [`run1`](raw/m5-webgpu-sft-dom-pilot-seed2027-run1.json), [`run2`](raw/m5-webgpu-sft-dom-pilot-seed2027-run2.json), [`run3`](raw/m5-webgpu-sft-dom-pilot-seed2027-run3.json) |
| Seed-2027 structured feature-materialization audit | Same SFT checkpoint and heads | [`sft-structured-context-robustness-seed2027.summary.json`](sft-structured-context-robustness-seed2027.summary.json) | Offline PyTorch audit, not a browser run |
| Seed-2027 full-stack corrected export parity | Same SFT checkpoint, fp32/fp16 graphs, serialized heads, and reused action suite | [`sft-structured-export-parity-seed2027.summary.json`](sft-structured-export-parity-seed2027.summary.json) | Offline CPU parity diagnostic, not a browser run or new capability test |
| Corrected trailing-compute browser arm | Same frozen SFT bundle | [`webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json`](webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json) | Superseded before external timestamp or browser runs; frozen hashes were not relabelled |

## BrowserGym/MiniWoB reset-goal acquisition

Two sequential controlled acquisitions ran the same pinned 60-task, 41-similarity-group plan
with seeds 11, 17, 23, and 29 in separate Python and Chromium processes. The resulting
240-row captures were byte-identical, as were their producer receipts. The capture contains
163 exact unique goal strings; normalized prompt deduplication also retains 163 prompts and
removes 77 duplicate rows. Goals vary across seeds for 45 tasks and remain static for 15.

The tracked
[reproducibility report](browsergym-miniwob-reset-capture-20260728.reproducibility.json) binds the
exact capture, receipt, source revisions, tracked runtime manifest, prompt export, adapter audit,
licenses, freeze contract, denylist, and provenance identities. An independent clean-directory
replay passed the public verifier and reproduced both the 163-record denylist and provenance
manifest byte-for-byte. The raw captures and receipts remain under the ignored
`data/private/browsergym/` boundary. A tracked test verifies the report and its canonical
self-hash without reading those private files.

This is a **reset-goal acquisition result only**. The producer copies
`observation.goal` returned by `reset`, takes no actions, and records no rewards, labels, episode
steps, or success outcomes. The prompts are inputs to corpus decontamination. This artifact is
not an episode-score result, agent evaluation, task-success measurement, or chronologically fresh
benchmark result.

## Mind2Web protected archive, bounded export, and prompt-only freeze

The exact pinned encrypted test archive was acquired from the official dataset repository:
567,745,122 bytes with SHA-256
`8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89`.
All 15 ZipCrypto/DEFLATE members were privately extracted, then the production adapter streamed
the archive plaintext and proved byte-for-byte equality with the 6,107,912,752 extracted bytes.
The tracked
[reproducibility report](mind2web-protected-archive-20260728.reproducibility.json) publishes only
permitted paths, counts, hashes, archive metadata, and evidence identities; protected prompts,
HTML, identifiers, labels, and actions remain private.

The first full-DOM v1 adapter run failed closed because one prompt was 858,832 bytes, above its
524,288-byte safety cap. That remains historical interface evidence, not the current blocker. The
production v2 label-blind ranker subsequently exported 9,378 prompt-only rows across 1,341 tasks
with a 1,771-byte maximum prompt. Independent raw-chain replay reproduced the v3 9,378-record
denylist byte-for-byte, and the completed four-suite aggregate manifest binds it.

These artifacts contain no labels or expected outputs and support only corpus decontamination—not
a Mind2Web score or chronological-freshness claim. Ranker recall was not measured and must be
reported as a native-evaluation ceiling. The published benchmark canaries were absent from the
protected plaintext members. The final corpus audit is limited to its explicitly supplied denylist
and is non-exhaustive, so no broader benchmark-canary guarantee is claimed.

## Trained one-TPP pretraining proxy

The bounded raw mixture contains 120,014,016 accepted characters in 24,125 documents and has
SHA-256
`22d4270ad6157a9701e86be8bfd73a4fc9c480dd2cfd82337a4d6a5218183e6c`. It comprises 78,002,243
FineWeb-Edu-Dedup characters, 18,001,053 Cosmopedia-v2 characters, and 24,010,720 permissively
licensed Python characters. Quality filtering retained 24,004 documents. The fixed
23,764/240-document train/validation split contains 28,045,897 training tokens and 287,995 packed
validation tokens; the scorecard covers 287,615 source tokens. The packed manifest SHA-256 is
`6a10cc606902a648258dc58ddb3ba19aa68c5b5ed6d812fcb2f06cbedfcaa9fd`, and the train-only
16,384-token BPE SHA-256 is
`8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c`.

Both arms used seed 2026, AdamW, the same WSD schedule and draws, 322 updates, and exactly
10,551,291 loss tokens. Because both arms use WSD, this is not a schedule comparison. The matched
validation scorecards are:

| Slice | Attention CE / BPB / top-1 | Hybrid CE / BPB / top-1 | Hybrid change |
|---|---:|---:|---:|
| Aggregate, 240 docs | 6.0547 / 2.0989 / 15.24% | 5.8617 / 2.0319 / 17.28% | CE −0.1930; BPB −0.0669; +2.04 pp |
| General, 216 docs | 6.1901 / 2.0785 / 14.39% | 6.0043 / 2.0161 / 16.23% | CE −0.1859; BPB −0.0624; +1.84 pp |
| Code, 24 docs | 5.3951 / 2.2204 / 19.39% | 5.1673 / 2.1266 / 22.42% | CE −0.2278; BPB −0.0937; +3.03 pp |

Ten thousand paired nonparametric document bootstraps give aggregate 95% intervals for
attention-minus-hybrid of `[0.1835, 0.2034]` CE, `[0.0634, 0.0707]` BPB, and
`[-2.271, -1.834]` top-1 percentage points. The general and code intervals also exclude zero.
These intervals condition on one architecture seed and the same 240 held-out documents; they are
not multi-seed architecture uncertainty.

The attention checkpoint SHA-256 is
`b86929f708b0294ff305fa9ffbfa5059e04a807facfc0c5c55d64c471215f4a9`; the hybrid checkpoint
SHA-256 is
`00dd2cf6651b0a27e18d707d287b464361e4f0636c7c787fafc7570682ab2e6d`. The summary binds exact
upstream revisions and license counts, configs, manifests, checkpoint/metrics/scorecard hashes,
and full-precision values.

This is one seed at approximately one loss token per parameter. The 4,854-entry denylist covers
only three local prompt exports and is bounded/non-exhaustive; BFCL, Mind2Web, WebLINX, and
BrowserGym exports were absent. The corpus has no structured-data source. The result supports no
architecture selection, tool-use, structured-output, browser-task, or agent claim.

## Trained pretrain-only cache-bearing result

The exact checkpoint hashes above were exported as parity-gated fp16 cache-bearing graphs and
measured in three page/session runs on one Apple M5 with Chrome 150 and ONNX Runtime Web 1.27.0.
Across both arms and four contexts, the artifact retains 720 measurements after 72 warmups.

| Input tokens | Hybrid p50 wall tokens/s | Hybrid p95 TPOT (ms) | Median joint gate |
|---:|---:|---:|---|
| 128 | 137.711 | 10.305 | fail |
| 512 | 128.232 | 9.610 | pass |
| 1,024 | 123.779 | 10.214 | fail |
| 1,536 | 116.476 | 10.205 | fail |

The hybrid cleared p50 wall throughput of at least 100 tokens/s at every context in every page
run. The joint gate also requires p95 TPOT at most 10 ms. Only the 512-token condition passes
using the median-of-run statistic, and no context passes both thresholds in every run; the overall
joint gate therefore fails.

The [trained summary](m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json) binds the
exact checkpoint, tokenizer, graph, runtime, and raw identities. The byte-for-byte payloads are
[run 1](raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run1.json),
[run 2](raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run2.json), and
[run 3](raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run3.json).

This browser artifact measures latency only. Language quality was scored separately and joined by
checkpoint hash; the payload itself contains no capability score. The checkpoints are pretrain
only, with no midtraining, SFT, RL, action, browser-task, or agent evaluation. The exact WebGPU
provider request succeeded, but adapter identity and per-node placement/fallback remain unknown.
Coverage is one M5, Chrome version, and runtime version—not multi-seed or cross-device evidence.

## Seed-2027 post-training pilot

The mechanically validated
[stage summary](webgpu-proxy-pilot-seed2027.summary.json) binds the provisional seed-2027 hybrid
pretrain checkpoint to 64 midtraining, 320 SFT, and 32 offline-RL steps. All three stages resolved
`device: auto` to MPS/fp32 under PyTorch 2.13.0; unlike the earlier pretraining runner, these
artifacts persist the resolved execution environment.

Summary schema v2 also embeds and re-verifies each stage's canonical config/data lineage, Git
commit/dirty/worktree identity, tokenizer, parent checkpoint, and per-input path/byte/SHA-256
records. Every availability label is `local_file_verified`: the summary commits to those local
blobs but does not claim that ignored checkpoints, graphs, or metrics are downloadable from a
fresh clone. An anonymous archival bundle remains required for reviewer reproduction.

On deterministic same-draw held-out inputs, midtraining changed agent loss from 7.7869 to 2.6371
and agent token accuracy from 3.71% to 69.80%. General held-out loss regressed slightly from
5.7064 to 5.7342 and token accuracy from 18.68% to 18.30%, so this is directional agent-domain
adaptation with a small general regression, not a clean promotion result. SFT reduced held-out
assistant loss from 2.7320 to 1.8146 and raised assistant-token accuracy from 67.29% to 73.13%;
teacher-forced all-assistant-token exactness moved from 0/65 to 1/65. This is not a free-running
generation metric.

The midtraining schedule's 25%→50% agent value was a row-draw weight, not a realized token share:
agent rows supplied 10,290/645,170 input tokens (1.60%) and 6,673/641,553 loss tokens (1.04%).
The pilot therefore does not estimate a 25–50% agent-token mixture.

The offline canonical-toolcall RL control encountered only six informative groups out of 128,
but it did realize 12 optimizer updates. Its 53-row greedy held-out evaluation was unchanged:
exact match remained 1/53, tool exact match 0/51, format validity 13/51, and mean reward 0.0434.
The zero delta is the result. Because RL updates only the autoregressive LM, its artifact
invalidates the SFT route/select/pointer heads; the browser pilot therefore uses the SFT
checkpoint, not the RL checkpoint.

## Internally prespecified pre-assistant-padding SFT WebGPU action and DOM pilots

The exact SFT checkpoint was exported to a 21,430,301-byte fp16 hidden-only action graph whose
PyTorch/ONNX parity gate passed. Three fresh Chrome 150 / ONNX Runtime Web 1.27.0 page sessions on
one Apple Metal 3 adapter each ran 20 held-out action cases 30 times under the internally
prespecified pre-assistant-padding stress condition at exactly 512 final tokenizer tokens.
Within-run TTFA p50
values were 24.75, 24.55, and 25.20 ms; p95 values were 34.405, 34.30, and 34.80 ms. The pooled
1,800-opportunity values were 24.80 ms p50 and 34.40 ms p95.

That speed did not translate into tool use. The policy predicted `null` for every case. The
capability denominator was 1/20 unique cases overall—0/19 tool-required and 1/1 abstention. Across
timing repetitions this becomes 90/1,800 exact and 0/1,710 tool-required; those rows are not
independent capability tasks. Exact action and `Success@100ms` through `Success@2s` were all 5%.
Schema validity was 100% only
because abstention is a valid action-suite output; it is not evidence of correct tool schemas.
This fixed-512 condition fails the capability gate.

Historical `rtab-0.2` action rows omitted normalized predicted arguments and the full expected
action, so their exact-action and schema booleans are explicitly labeled
`browser_reported_non_recomputable` in the regenerated summary. The all-abstention tool-name
failure is directly visible, but argument-level rescoring is unavailable. Historical DOM rows do
retain the raw action and validator schema, and their scores are independently recomputed.

The matched local DOM harness strengthens the negative result. All 8 unique tool-required tasks
failed; across three 240-row timing runs, all 720 opportunities abstained. Exact action,
independent executable-schema validity,
final DOM state, and closed-loop success were all 0%. Pooled closed-loop latency was 33.30 ms p50
and 66.80 ms p95, but there were no useful actions. The
[DOM summary](m5-webgpu-sft-dom-pilot-seed2027.summary.json) mechanically reproduces the three
raw payloads.

An exploratory offline parity diagnostic separates model capability from the benchmark's feature
materialization. At natural prompt lengths, the held-out action set routes 17/20 cases correctly
and the selector is top-1 correct on 17/19 tool cases. The natural action set produces zero text
routes, so it misses the sole abstention; on the independent frozen eval decisions, natural route
accuracy is 83/98, selector top-1 is 72/79, and dispatched tool accuracy is 70/79. The
internally prespecified stress condition inserts real, unmasked single-token spaces immediately
before the assistant marker and reads the final hidden state; it collapses the route to text for
every case
at 128 tokens and above. Native PyTorch, fp32/fp16 ONNX, and exported JSON heads agree, so export
or fp16 conversion is not the cause. This diagnostic is not a natural-context WebGPU quality
measurement and does not rescue the failed fixed-512 result.

A corrected fixed-compute runner now appends filler after the natural assistant marker, still runs
the 512-token graph, dispatches from `hidden[natural_input_tokens - 1]`, and restricts pointer
scans to the natural token span. The
[offline audit](sft-structured-context-robustness-seed2027.summary.json) preserves natural and
trailing-compute metrics exactly on both frozen suites, and the
[corrected browser protocol](webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json) locally
froze runner and artifact identities, but the runner was intentionally upgraded to the
cache-bearing autoregressive ABI before an external timestamp or any corrected-condition run.
That protocol is now superseded; its hashes were not rewritten, and any future run requires a new
freeze against the final trained bundle and current runner.
Because the action, DOM, and 65-row suites informed the diagnosis and runner correction, those
measurements are reused-suite deployment-parity re-evaluations, not confirmatory capability.

The [full-stack export-parity gate](sft-structured-export-parity-seed2027.summary.json) executes
the exact browser grounding and normalization functions under Node and compares native PyTorch
with the fp32/fp16 ONNX graphs plus serialized heads at the corrected 512-token decision index.
Both ONNX variants agree exactly on 20/20 routes, tools, grounded arguments, and normalized
actions, including 11/11 learned pointer spans. The shared offline diagnostic score is 16/20 exact
and 20/20 schema-valid, but the suite was reused during diagnosis; these are parity diagnostics,
not a new independent capability estimate or a WebGPU measurement.
If the paper instead claims genuine
capability under the original pre-marker 512-token materialization, the heads must be trained and
evaluated on that exact input condition.

Missing comparisons remain material: the same action-trained checkpoint has not been measured
with cache-bearing autoregressive JSON or candidate-trie controls; BrowserGym/open-web,
cross-device, the 34M at-least-five-TPP architecture screen, its promoted 20-TPP/downstream
comparison, and natural-prompt browser evaluations are still uncollected.

## Random-weight cache-bearing sizing result

Each cache condition performs one prompt-length prefill graph pass and 31 calls to a separate
decode graph whose token axis is fixed at `T=1`. The exporter hard-gates the artifacts before
browser use over multiple prompt lengths and multiple decode steps: greedy next-token IDs agree
exactly for ONNX versus the PyTorch cached path and for the PyTorch cached path versus fresh
full-context execution, while floating cache tensors remain within the declared precision-specific
tolerance.

The table reports the median of three within-run p50 wall-decode rates; samples are not pooled
across page/session runs.

| Hybrid pair | 128 tokens | 512 tokens | 1,024 tokens | 1,536 tokens | 100 tok/s engineering reference |
|---|---:|---:|---:|---:|---|
| 34.2M | 74.05 | 64.54 | 57.53 | 47.15 | misses at all contexts |
| 15.6M | 90.87 | 89.26 | 80.94 | 79.77 | misses at all contexts |
| 10.5M | 159.23 | 160.46 | 143.49 | 127.57 | clears at all contexts |

The 34.2M hybrid is faster than its parameter-matched all-attention control at all four contexts,
but it still misses 100 tok/s. The 100 tok/s line is an engineering reference for this latency
screen, not an action deadline, quality threshold, or claim about useful generated text.

For the measured WebGPU condition, the page requested `gpu-buffer` output location for each
`present_*` cache and CPU location for `next_token`; every decode pass reported those cache tensors
as `gpu-buffer`. The page rebound each present tensor directly as the next call's corresponding
past input without reading cache contents into JavaScript, then disposed superseded and final
tensors. This is bounded runtime evidence, not proof of physical GPU placement: ONNX Runtime Web
did not expose adapter identity, per-node placement, or per-node fallback status.

The graph creates fresh presents on every call. Attention K/V grows through append/concat and
short-conv state is replaced by a fresh fixed-width tail; this is not an in-place or paged-cache
implementation. Accordingly, these measurements include that allocation/update strategy.

The cache-bearing graphs are standalone latency artifacts. They are not integrated into the
trained complete-action runner. The two autoregressive controls in that runner still recompute the
entire prefix for every output token and report `decode_cache: false` with
`decode_strategy: "full_context_recompute"`.

## Integrity and training status

Each summary declares the sizes and SHA-256 hashes of its raw payloads and recomputes the reported
within-run percentiles, run ranges, and paired ratios from those records. The tracked result tests
independently verify those declarations and calculations.

The large ONNX graphs are intentionally not tracked. The 34M hidden-only fp16 graphs are about
66 MiB each; one 34M cache-bearing prefill or decode graph is about 80 MiB, and the smaller-pair
graphs are smaller. Their hashes, exact model configs, parameter counts, export contract, and
parity records remain content-bound in the raw payloads and matched-export manifests.

The 10.5M pretraining and post-training pilots above do not satisfy the paper selection or
capability gates. The 34M matched five-tokens-per-parameter architecture screen is staged but has
not started; it is a promotion screen, not the final quality claim:
seed 2026 ([hybrid](../../../configs/train/pretrain-paper-5tpp-hybrid-seed2026.yaml),
[attention](../../../configs/train/pretrain-paper-5tpp-attn-seed2026.yaml)), seed 2027
([hybrid](../../../configs/train/pretrain-paper-5tpp-hybrid-seed2027.yaml),
[attention](../../../configs/train/pretrain-paper-5tpp-attn-seed2027.yaml)), and seed 2028
([hybrid](../../../configs/train/pretrain-paper-5tpp-hybrid-seed2028.yaml),
[attention](../../../configs/train/pretrain-paper-5tpp-attn-seed2028.yaml)). Until those runs and
their held-out gates complete, the latency screens do not select a trained architecture.

# Weight-transfer analysis

`scripts/analyze_weight_transfer.py` measures how a child checkpoint relates to a pretrained or
parent checkpoint.  It compares model configuration and tokenizer identity, verifies shared tensor
shapes, reports L2 movement/cosine per tensor and per parameter family, and lists newly introduced
tool/pointer/route/selector heads.

Example:

```bash
PYTHONPATH=src python scripts/analyze_weight_transfer.py \
  --base runs/pretrain-webgpu-proxy-1tpp-hybrid-seed2027/latest.pt \
  --target runs/midtrain-webgpu-proxy-pilot-hybrid-seed2027/latest.pt \
  --out /tmp/pretrain-midtrain-transfer.json
```

## Existing lineage measurement

The report was run on the local checkpoints that already carry stage lineage:

| Transition | Shared tensors | Config/tokenizer | Largest movement |
| --- | ---: | --- | --- |
| pretrain → midtrain | 40 | identical / identical | embedding relative ΔL2 = **0.0615** |
| midtrain → SFT | 40 (+ 11 new action-head tensors) | identical / identical | embedding relative ΔL2 = **0.4054** |
| SFT → AndroidControl pilot | 40 (+ 11 retained action-head tensors) | identical / identical | embedding relative ΔL2 = **0.00140** |
| SFT → AndroidControl+AITW mixture pilot | 51 (40 backbone + 11 retained heads) | identical / identical | embedding relative ΔL2 = **0.00136** |

For pretrain → midtrain, the relative ΔL2 values were 0.0266 for attention/mixer, 0.0337 for
FFN, and 0.0041 for normalization.  For midtrain → SFT they were 0.0346, 0.0431, and 0.0095.
These are movement measurements, not accuracy improvements.  The SFT checkpoint adds
`tool_head`, `ptr_head`, `route_head`, and `dense_selector` tensors; they do not exist in the
midtrain state dictionary and therefore must be initialized by a controlled seed.

The bounded AndroidControl continuation changed only the shared backbone: action heads were
retained unchanged (relative ΔL2 = 0), while attention/mixer, FFN, embedding, and normalization
movement were 0.00099, 0.00116, 0.00140, and 0.000035 respectively.  This is consistent with a
small learning-rate continuation and is not evidence that mobile actions were learned; the pilot
had no exact trajectory matches and no emulator reward.

The 26-row AndroidControl+AITW mixture shows the same transfer pattern: action heads remained
unchanged, while attention/mixer, FFN, embedding, and normalization moved by 0.00094, 0.00110,
0.00136, and 0.000035 respectively (51 shared tensors; identical config and tokenizer).  The
first mixture receipt was later invalidated because its pilot loaded a byte tokenizer against the
BPE parent; its 2.02% assistant-token accuracy and 0% exact trajectories are retained only as a
diagnostic.  The corrected BPE run (8 updates, same 26 rows) reports `25.68%` → `31.69%`
assistant-token accuracy and `5.4868` → `4.7944` mean loss, with exact trajectories still `0/26`.
These are compatibility and language-model bridge measurements, not mobile-control evidence.

The corrected v2 continuation (200 SFT updates after moving the action instruction to the right
edge of the projected observation) was also measured from the same 26-row parent. It preserved all
51 tensor shapes, config, and tokenizer identity; action heads remained frozen. Relative movement
was `0.01404` for attention/mixer, `0.01650` for FFN, `0.03290` for the embedding, and `0.000853`
for normalization. Mean normalized-row loss fell from `7.9735` to `3.3477`, with assistant-token
accuracy `75.93%`, but exact assistant sequences remained `0%`.

The old dispatch pilot's held route `1.0` / mobile selector `0.4` report is superseded: that run
loaded the default byte tokenizer against the BPE checkpoint.  The corrected v6 run loads the
checkpoint-recorded BPE tokenizer and reaches productivity train selector `1.0`, productivity
held selector `0.75`, and productivity held route accuracy `1.0`; broader mobile held selector is
still `0.4`.  The v6 browser receipt still uses the explicit mobile lexical guard for seven mobile
rows, so the learned no-guard mobile policy remains unproven.

The v6 transfer audit confirms exact compatibility: 51 shared tensors, no config mismatch, equal
tokenizer SHA-256, and no shape additions/removals.  The training probe intentionally froze the
backbone, so attention/mixer, FFN, embedding, and normalization movement is `0.0`; the action-head
group moved by aggregate relative ΔL2 `1.7017` (mostly the dense selector at `1.9213`).  This is
the expected head-only adaptation pattern, not evidence that the pretrained representation is
optimal.  The low-rate-unfrozen and no-transfer comparison is recorded below.

The v9 selector continuation adds 86 instruction-only mobile paraphrases (repeated four times for
the probe) and removes inherited full accessibility dumps from the mobile tool centroids.  It
raises the external mobile held selector from 4/10 to 6/10 while preserving 100% route accuracy;
productivity held selector remains 3/4.  Its transfer audit still finds 51 shared tensors, equal
config/tokenizer identity, zero backbone movement, and action-head aggregate relative ΔL2 `1.6521`.
The v9r retrieval sidecar is deliberately excluded from these neural weight claims because it has
no trainable parameters.

The v14 state-conditioned continuation provides the next controlled probe comparison.  It starts
from the v11 parent with the same 10.5M BPE model and freezes all 40 backbone tensors while adding
23 disjoint state/browser examples to the route and dense-selector probe pool.  The shared-backbone
relative ΔL2 is `0.0`; route-head and dense-selector relative ΔL2 are `0.2548` and `0.2580`.  Train
selector accuracy rises to `89.66%`, while held-out mobile/productivity selector accuracy remains
`60%`/`75%`.  The guarded WebGPU stateful gate reaches `4/13` exact actions and `3/13` closed-loop
steps, with complete-trajectory pass@1 still `0/3`.  This is evidence for cheap probe transfer and
its current limit, not evidence that the frozen representation is sufficient for browser-agent
control.  The full v14 artifact and first-failure receipt is
[`m16-webgpu-mobile-productivity-trajectories-v14.json`](paper/results/raw/m16-webgpu-mobile-productivity-trajectories-v14.json).

The m56 stateful-productivity transfer ablation closes the proposed comparison on one fixed local
state machine.  It uses the same 62-tool pool, 16 held-out decisions, seed, and pointer budget for
three arms: frozen pretrained, pretrained with a `1e-5` backbone learning rate, and matched random
backbone.  The low-rate arm improves selector top-1 from `53.33%` to `73.33%` over the frozen arm
and reaches `73.33%` versus `33.33%` for random; its aggregate backbone relative ΔL2 is `0.188%`
(mixer `0.411%`, FFN `0.409%`, embedding `0.046%`, normalization `0.0084%`).  Closed-loop
success remains `1/16` for every arm, complete workflows remain `1/5` (abstention only), and
recovery remains `0/1`.  The result supports low-rate pretrained initialization as a selector
optimization prior, but does not support adopting it as evidence of stateful execution.  See the
[`m56 receipt`](paper/results/raw/m56-stateful-productivity-transfer-ablation-v1.json) and
[`low-rate trainer`](../scripts/train_stateful_productivity_lowrate_probe.py).

The m142 five-surface continuation repeats the low-rate transfer measurement after adding public
AndroidControl, AgentNet, Mind2Web, ToolSandbox, and redacted MCPMark projections to one
source-disjoint mixture.  The 10.5M BPE parent is configuration- and tokenizer-identical to the
child; the 48-update child improves aggregate held-out assistant-token accuracy from `61.15%` to
`68.65%`, but exact sequences remain `0%` and MCPMark falls slightly (`26.47% → 26.31%`).
The matched random-backbone control reaches only `39.68%` aggregate token accuracy under the
same schedule, a warm-minus-random gap of `+28.97` points.  Backbone relative movement remains
small (embedding `0.742%`, FFN `0.325%`, mixer `0.255%`, normalization `0.016%`) and action heads
are unchanged.  This is a stronger cross-surface
compatibility/transfer audit, not evidence for a universal agent head or native tool execution;
the full source hashes and decision boundary are in the [`m142 receipt`](paper/results/raw/m142-five-surface-public-continuation-v1.json).

The follow-up m143 native ToolSandbox audit applies the same warm child and matched random
control to all 129 public base scenarios under the pinned simulator and milestone verifier.  Both
arms score `28/129` (`21.71%`) under the deterministic one-step scripted-user protocol, so this
native diagnostic shows no aggregate warm-start advantage.  It is still useful as a negative
control: native execution can run without exceptions, but the current transfer does not establish
stateful tool-use quality, and the run is not official-split eligible because the model-based user
simulator and full augmented matrix were not executed.  See the [`m143 receipt`](paper/results/raw/m143-toolsandbox-native-base-transfer-audit-v1.json).

The AgentNet/OpenCUA adapter is deliberately outside this weight-transfer series.  Its 12-action
official sample is parsed into evaluation-only coordinate tools, so it changes no checkpoint
weights and cannot be counted as a training gain.  A screenshot/accessibility grounding bridge
and the official AgentNetBench prediction protocol are required before a WebGPU model can be
compared on that benchmark.

The m59 ToolSandbox public-source projection provides a sharper small-data control.  Static-AST
rows from the pinned scenario definitions improve the transferred child's held-out token accuracy
from `65.31%` to `71.53%` after 32 updates, with backbone relative movement of `0.196%` in
attention/mixer, `0.476%` in embeddings, and `0.243%` in FFN weights.  A candidate-list selector
probe improves the inherited selector from `45%` to `75%` top-1, but a matched random backbone
also reaches `75%`; selector movement is large (query `1.321`, tool `1.035` relative ΔL2).  This
supports retraining the tool-specific head while rejecting the stronger claim that the pretrained
representation itself is responsible for the gain.  See the [`m59 receipt`](paper/results/raw/m59-toolsandbox-public-projection-transfer-v1.json)
and [`ToolSandbox adapter`](../scripts/ingest_toolsandbox_public.py).

The first bounded public browser continuation is recorded in
[`mind2web-public-train-sample-v1.json`](paper/results/raw/mind2web-public-train-sample-v1.json).
Sixteen updates on 10 normalized Mind2Web training trajectories moved the transferred backbone by
relative ΔL2 `0.00162` (attention/mixer), `0.00189` (FFN), `0.00260` (embedding), and `0.000062`
(normalization); action heads remained unchanged.  Config and tokenizer identity matched across
all 51 shared tensors.  Mean loss fell `2.2790` → `1.7595` and assistant-token accuracy rose
`63.97%` → `74.29%`, but exact trajectories stayed `0/10`.  The evidence supports reusing the
verified BPE parent with a lower-rate backbone continuation for browser adaptation; it does not
support publishing the child as a Mind2Web result or claiming that these movement magnitudes are
universally optimal.

## Adoption protocol

1. **Compatibility gate:** require identical model config fields and tokenizer SHA-256.  Refuse
   shape mismatches rather than partially copying a tensor.
2. **Backbone transfer:** copy only shared same-shape tensors from the verified parent checkpoint.
   Keep the checkpoint's lineage and parent SHA-256 in the child receipt.
3. **Head initialization:** initialize action heads independently and record their seed.  Never
   mistake a new head's presence for learned capability.
4. **Optimization groups:** use a lower learning rate for the transferred backbone and a higher
   rate for fresh heads.  The exact ratio is an experiment variable; compare it with a no-transfer
   baseline and a uniform-rate baseline.
5. **Ablation and acceptance:** accept transfer only if it improves the pre-registered held-out
   metrics without increasing abstention, contamination, latency, or WebGPU memory.  The movement
report alone cannot establish that claim.

This protocol is compatible with the existing strict parent-checkpoint checks in
`src/localagent/train/stage_data.py` and `src/localagent/train/update_preflight.py`.

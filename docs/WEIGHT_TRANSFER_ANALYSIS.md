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
mixture's held-row assistant-token accuracy was 2.02% with 0% exact trajectories, so these small
movements establish lineage and compatibility only; they do not justify claiming learned mobile
control or selecting this checkpoint without a held-out ablation.

The corrected v2 continuation (200 SFT updates after moving the action instruction to the right
edge of the projected observation) was also measured from the same 26-row parent. It preserved all
51 tensor shapes, config, and tokenizer identity; action heads remained frozen. Relative movement
was `0.01404` for attention/mixer, `0.01650` for FFN, `0.03290` for the embedding, and `0.000853`
for normalization. Mean normalized-row loss fell from `7.9735` to `3.3477`, with assistant-token
accuracy `75.93%`, but exact assistant sequences remained `0%`. The dispatch child then achieved
held route accuracy `1.0` and held mobile selector top-1 `0.4` (10 held turns; train top-1 `0.5385`).
These are useful transfer diagnostics, not a publishable mobile policy: the browser receipt's 7/7
mobile actions are from the explicit lexical guard, while the learned selector still fails the
no-guard acceptance criterion.

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

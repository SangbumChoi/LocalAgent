# Deploy the latest model + WebGPU demo to Hugging Face

Everything here is **prepared but not pushed** — publishing to your HF account is yours to run.
Requires `huggingface_hub` (installed) and a write token. Replace `SangbumChoi` with your namespace.

The deployable checkpoint is `runs/tiny-30m-scenarios-best.pt` (free-form OOD call-name ~57%,
multi-turn next-tool selection ~74%). The app (`app.js`) is already rewritten for the generable
dispatch (route head → dense selector → pointer-copy over the 50-tool surface).

## 0. Authenticate (once)
```bash
hf auth login          # paste a token with write access  (or: export HF_TOKEN=hf_xxx)
```

## 1. Export the inference bundle from the latest checkpoint
```bash
python -c "from localagent.inference.export.to_onnx import export_web; \
           export_web('runs/tiny-30m-scenarios-best.pt', 'build/web', action_only=True)"
# writes the full logits graph, hidden-only action graph, heads/meta, and bundle-manifest.json
# bundle-manifest.json is published only after all fp32/fp16 graphs pass hard PyTorch parity

python -c "import json, pathlib; p=pathlib.Path('build/web'); \
           m=json.loads((p/'meta.json').read_text()); \
           b=json.loads((p/'bundle-manifest.json').read_text()); \
           assert m.get('action_model_file') == 'action_model.fp16.onnx'; \
           assert b['schema_version'] >= 3; \
           assert b['parity_gate']['hard_gate'] and b['parity_gate']['passed']; \
           assert b['parity_gate']['results'][m['action_model_file']]['passed']"
```

## 2. Model repo — host the checkpoint + ONNX
```bash
hf repo create danelcsb/localagent-tiny-30m-byte --repo-type model -y || true
hf upload danelcsb/localagent-tiny-30m-byte runs/tiny-30m-scenarios-best.pt model.pt        --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/model.onnx           model.onnx      --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/model.fp16.onnx      model.fp16.onnx --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/action_model.onnx action_model.onnx --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/action_model.fp16.onnx action_model.fp16.onnx --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/dispatch_heads.json  dispatch_heads.json --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/bundle-manifest.json bundle-manifest.json --repo-type model
# (load in PyTorch via this repo's LocalAgentLM/ModelConfig — pure PyTorch, no transformers dep)
```

## 3. Space — the WebGPU demo
The Space is `sdk: static` (see the frontmatter in `README.md`). Copy the complete checked bundle
next to the app, then push the whole folder.
```bash
cp build/web/{model.fp16.onnx,action_model.fp16.onnx,heads.json,meta.json,dispatch_heads.json,bundle-manifest.json} spaces/localagent-webgpu/
hf repo create danelcsb/localagent-webgpu --repo-type space --space_sdk static -y || true
hf upload danelcsb/localagent-webgpu spaces/localagent-webgpu/ . --repo-type space \
  --exclude "DEPLOY.md"        # DEPLOY.md is for maintainers, not the live Space
```
`hf upload` puts the ONNX graphs on LFS automatically. The demo is then live at
`https://huggingface.co/spaces/danelcsb/localagent-webgpu`.

## Notes
- The bundle files are git-ignored deploy artifacts — they are NOT in this source tree; step 1
  regenerates them deterministically from the checkpoint.
- To verify locally before pushing: `cd spaces/localagent-webgpu && cp ../../build/web/*.{onnx,json}
  . && python -m http.server 8000` then open http://localhost:8000. The interactive demo may retry
  on WASM. Benchmark pages require an explicit provider, the manifest, and the distinct hidden-only
  action graph; they fail rather than falling back or relabeling the full logits graph.
- Suggested model-card blurb for the model repo README:
  > 28M-param, byte-level, pretrained-from-scratch tool-calling agent. Generable dispatch:
  > 5-way route head → dense two-tower selector (scores any tool by its description embedding) →
  > pointer-copy arguments. 50-tool surface; ~57% free-form OOD call-name, ~74% multi-turn next-tool
  > selection. Runs in-browser and requests ONNX Runtime Web's WebGPU provider — see the linked
  > Space. Per-node placement is not exposed by ORT Web.

## Matched cached-decode latency page

The cache-bearing benchmark is a separate untrained-random-weight latency artifact. It is not the
trained action bundle above. Export produces separate prefill and fixed-`T=1` decode graphs and
publishes `matched-decode.json` only after multi-length, multi-step token/cache parity passes.
Generate the 34.2M pair from the repository root:

```bash
python scripts/export_matched_webgpu_decode.py \
  --out runs/webgpu/random-cached-decode-latency-seed-20260728-v2
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/spaces/localagent-webgpu/decode-benchmark.html?backend=webgpu&manifest=../../runs/webgpu/random-cached-decode-latency-seed-20260728-v2/matched-decode.json
```

The two smaller matched pairs use the same exporter and page:

```bash
python scripts/export_matched_webgpu_decode.py \
  --hybrid-config configs/model/webgpu-16m-hybrid.yaml \
  --attention-config configs/model/webgpu-16m-attn.yaml \
  --out runs/webgpu/random-cached-decode-16m-seed-20260728
python scripts/export_matched_webgpu_decode.py \
  --hybrid-config configs/model/webgpu-10m-hybrid.yaml \
  --attention-config configs/model/webgpu-10m-attn.yaml \
  --out runs/webgpu/random-cached-decode-10m-seed-20260728
```

Use `backend=wasm` for the separately labeled WASM condition. The page is run-once: reload before
collecting another repetition set. If publishing this benchmark in a static Space, copy the entire
export directory (pair manifest, both provenance/config files, and graph files) under the Space
directory and pass its deployed `matched-decode.json` through the `manifest` query parameter. Do
not publish a graph without its exporter-produced manifest and provenance; the page fails closed
on their labels, trajectory-parity evidence, byte counts, and SHA-256 identities.

For WebGPU, the page requests `gpu-buffer` present-cache outputs and rebinds the returned tensors
as the next call's past inputs without reading cache contents into JavaScript. That is not evidence
of physical residency or per-node placement, which ONNX Runtime Web does not expose. The graph
returns fresh presents each step: attention uses append/concat and short-conv replaces a fresh
fixed-width tail. It is not an in-place or paged-cache implementation.

The tracked three-run summaries and every raw/config link are in the
[paper result index](../../docs/paper/results/README.md). Only the 10.5M hybrid clears the
100 tok/s engineering reference at all four tested contexts; the 34.2M and 15.6M pairs do not.
This latency page remains a separately labeled systems experiment. The trained complete-action
runner uses the same strict `next_token`, `[B,V]` logits, and cache ABI, but requires a final-RL
checkpoint lineage plus the exact `openai_full_catalog_v1` tokenizer/catalog contract.

## Production cached autoregressive bundle

Export the autoregressive control from the same final RL checkpoint and verified BPE tokenizer
with `export_cached_decode(...)`, including the non-empty `training_artifact_sha256` set. Copy its
complete output directory under `spaces/localagent-webgpu/cached/`; do not rename individual
graphs or sidecars. The browser defaults to `cached/meta.json` and loads this bundle lazily only
when an autoregressive policy is selected, so the structured action graph remains independent.

The deployed primary `bundle-manifest.json` must also contain an artifact identity whose
`file` is exactly `cached/provenance.json`, with the fetched file's byte count and lowercase
SHA-256. Benchmark-grade pages reject an unpinned cached provenance file. That provenance must in
turn pin `meta.json`, `training-lineage.json`, the copied tokenizer/config, and both selected
prefill/decode graphs. The runtime validates the complete chain before constructing either cached
ORT session; stale output names, dtypes, cache geometry, tokenizer/catalog metadata, checkpoint
ancestry, or graph identities stop the autoregressive policy instead of falling back to
full-context recomputation.

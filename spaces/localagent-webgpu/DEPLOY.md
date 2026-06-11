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
           export_web('runs/tiny-30m-scenarios-best.pt', 'build/web')"
# writes build/web/{model.onnx, model.fp16.onnx, heads.json, meta.json, dispatch_heads.json}
# (parity-checked vs PyTorch: route-head & dense-selector argmax/top-1 100% agreement)
```

## 2. Model repo — host the checkpoint + ONNX
```bash
hf repo create danelcsb/localagent-tiny-30m-byte --repo-type model -y || true
hf upload danelcsb/localagent-tiny-30m-byte runs/tiny-30m-scenarios-best.pt model.pt        --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/model.onnx           model.onnx      --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/model.fp16.onnx      model.fp16.onnx --repo-type model
hf upload danelcsb/localagent-tiny-30m-byte build/web/dispatch_heads.json  dispatch_heads.json --repo-type model
# (load in PyTorch via this repo's LocalAgentLM/ModelConfig — pure PyTorch, no transformers dep)
```

## 3. Space — the WebGPU demo
The Space is `sdk: static` (see the frontmatter in `README.md`). Copy the bundle next to the app,
then push the whole folder. `app.js` fetches the four bundle files relative to the page.
```bash
cp build/web/{model.fp16.onnx,heads.json,meta.json,dispatch_heads.json} spaces/localagent-webgpu/
hf repo create danelcsb/localagent-webgpu --repo-type space --space_sdk static -y || true
hf upload danelcsb/localagent-webgpu spaces/localagent-webgpu/ . --repo-type space \
  --exclude "DEPLOY.md"        # DEPLOY.md is for maintainers, not the live Space
```
`hf upload` puts `model.fp16.onnx` (~57 MB) on LFS automatically. The demo is then live at
`https://huggingface.co/spaces/danelcsb/localagent-webgpu`.

## Notes
- The bundle files are git-ignored deploy artifacts — they are NOT in this source tree; step 1
  regenerates them deterministically from the checkpoint.
- To verify locally before pushing: `cd spaces/localagent-webgpu && cp ../../build/web/*.{onnx,json} .
  && python -m http.server 8000` then open http://localhost:8000 (needs a WebGPU-capable browser;
  falls back to WASM otherwise).
- Suggested model-card blurb for the model repo README:
  > 28M-param, byte-level, pretrained-from-scratch tool-calling agent. Generable dispatch:
  > 5-way route head → dense two-tower selector (scores any tool by its description embedding) →
  > pointer-copy arguments. 50-tool surface; ~57% free-form OOD call-name, ~74% multi-turn next-tool
  > selection. Runs in-browser (WebGPU) — see the linked Space.

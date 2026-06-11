---
title: LocalAgent Tool Calling (WebGPU)
emoji: 🛠️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: Sub-100M from-scratch tool-calling agent in the browser
---

# LocalAgent — tool calling in the browser (WebGPU)

A **28M-parameter, pretrained-from-scratch** byte-level agent that does **grounded tool
calling** and **multi-step planning** — running **entirely in your browser** on
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) with the **WebGPU** backend
(WASM fallback when WebGPU is unavailable). No server, no API key; the model is downloaded once
and cached.

Model: [`SangbumChoi/localagent-tiny-30m-byte`](https://huggingface.co/SangbumChoi/localagent-tiny-30m-byte).
Source: [LocalAgent](https://github.com/sangbumchoi/localagent).

## What it shows (generable dispatch — no fixed-N classifier)

- **Route gate** — a 5-way head (`web_search / computer_use / code / app_action / text`) on the ONNX
  `hidden` output; the `text` route is **abstention** (answer directly / no tool).
- **Tool selection** — a **dense two-tower selector**: the query tower projects `hidden`, scored by
  cosine against a precomputed per-tool description-embedding matrix over the **50-tool** surface
  (`argmax_j q·tool_matrix[j]`). Adding/removing a tool is adding/removing a row — no retraining.
- **Grounded arguments** — copied from spans of your prompt via the learned pointer head, so the
  emitted call is schema-valid by construction.
- **Multi-step plans** — the rollout: pick a tool → ground it → feed back a simulated response →
  pick the next, until the route head emits `text`.

## How it runs (honest version)

The transformer forward pass runs on **WebGPU** via an exported ONNX graph that emits `logits` and
the last `hidden` state. The **route head**, the **dense selector** (matmul + normalize + argmax over
the precomputed tool matrix), the **pointer-copy** grounding, and the **planner loop** are light
JavaScript on top — a faithful port of the Python `routes` / `dense_selector` / `pointer_head`
pipeline (parity-checked at export: 100% argmax/top-1 agreement). First load fetches
`model.fp16.onnx` (~57 MB) and caches it.

## Files
- `index.html` / `style.css` — the UI shell.
- `app.js` — byte tokenizer, onnxruntime-web session (WebGPU + WASM fallback), route+selector dispatch,
  grounding, and the planner rollout.
- `model.fp16.onnx`, `heads.json`, `meta.json`, `dispatch_heads.json` — the exported inference
  bundle (**not in the source repo**; deploy artifacts). See `DEPLOY.md` for the exact commands.

## Deploy
See **`DEPLOY.md`** for copy-paste build + push commands. In short: export the bundle from the latest
checkpoint and upload the static app + the four bundle files into a `sdk: static` Space:

```bash
python -c "from localagent.inference.export.to_onnx import export_web; \
           export_web('runs/tiny-30m-scenarios-best.pt', 'build/web')"
```

`app.js` fetches `model.fp16.onnx` / `heads.json` / `meta.json` / `dispatch_heads.json` relative to
the page, so they must sit next to `index.html`. Export is parity-checked vs PyTorch (max |Δlogits|
7.6e-6; route-head & dense-selector argmax/top-1 100% agreement). The graph is standard opset-17;
onnxruntime-web falls back per-op to WASM for any op without a WebGPU kernel, with identical results.

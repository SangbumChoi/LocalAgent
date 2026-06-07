---
title: LocalAgent Tool Calling (WebGPU)
emoji: 🛠️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: A <100M from-scratch agent doing grounded tool calls in your browser
---

# LocalAgent — tool calling in the browser (WebGPU)

A **28M-parameter, pretrained-from-scratch** byte-level agent that does **grounded tool
calling** and **multi-step planning** — running **entirely in your browser** on
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) with the **WebGPU** backend
(WASM fallback when WebGPU is unavailable). No server, no API key; the model is downloaded once
and cached.

Model: [`SangbumChoi/localagent-tiny-30m-byte`](https://huggingface.co/SangbumChoi/localagent-tiny-30m-byte).
Source: [LocalAgent](https://github.com/sangbumchoi/localagent).

## What it shows

- **Tool selection** — the model's *real* `tool_head` decision (a linear head on the ONNX
  `hidden` output) over the 21-tool surface, with a confidence score, plus **abstention** when no
  tool fits.
- **Grounded arguments** — arguments copied from spans of your prompt, so the emitted call is
  schema-valid by construction.
- **Multi-step plans** — the learned `plan_rollout`: pick a tool → ground it → feed back a
  simulated response → pick the next, until the model emits the *stop* (`text`) class.

## How it runs (honest version)

The transformer forward pass runs on **WebGPU** via an exported ONNX graph that emits both `logits`
and the last `hidden` state. The **tool head** (one matmul + argmax over `hidden`), the
**argument grounding**, and the **planner loop** are light JavaScript on top — a faithful port of
the Python `tool_head` / grounding / `plan_rollout`. Arg grounding in-browser covers the common
formats (paths, URLs, quoted strings, names, numbers); the full Python grounder is the source of
truth. First load fetches `model.fp16.onnx` (~tens of MB) and caches it.

## Files
- `index.html` / `style.css` — the UI shell.
- `app.js` — byte tokenizer, onnxruntime-web session (WebGPU + WASM fallback), tool selection,
  grounding, and the planner rollout.
- `model.fp16.onnx`, `heads.json`, `meta.json` — the exported inference bundle (**not in the
  source repo**; they are deploy artifacts).

## Deploy
The model bundle is produced from a trained checkpoint, separately from the source tree:

```bash
python -c "from localagent.inference.export.to_onnx import export_web; \
           export_web('runs/tiny-30m-byte-best.pt', 'runs/web_export')"
```

Then upload **the four static files + the three bundle files** into a Hugging Face Space repo
(`sdk: static`), all at the repo root, using git-lfs for the large ones:

```bash
huggingface-cli upload <user>/localagent-webgpu spaces/localagent-webgpu/ . --repo-type space
huggingface-cli upload <user>/localagent-webgpu runs/web_export/model.fp16.onnx model.fp16.onnx --repo-type space
huggingface-cli upload <user>/localagent-webgpu runs/web_export/heads.json heads.json --repo-type space
huggingface-cli upload <user>/localagent-webgpu runs/web_export/meta.json meta.json --repo-type space
```

`app.js` fetches `model.fp16.onnx` / `heads.json` / `meta.json` relative to the page, so they must
sit next to `index.html`. The export was verified for onnxruntime-CPU↔PyTorch parity
(max |Δlogits| 7.6e-6; fp16 drift 1.4e-3, same tool argmax) and the in-browser tool-selection +
pointer-grounding math was checked against the bundle (`get_weather{city:"Paris"}`,
`read_file{path:"tests/test_api.py"}`, …). The graph is all standard opset-17 ops; onnxruntime-web
falls back per-op to WASM for any op without a WebGPU kernel (`Trilu`/`Tile`/`Expand`), with
identical results.

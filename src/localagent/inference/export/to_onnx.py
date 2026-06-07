"""Export the model to ONNX (Phase 9) — enables onnxruntime / onnxruntime-web (WASM + WebGPU).

Exports the full-sequence forward (logits over a byte prompt); browser runtimes
(onnxruntime-web, transformers.js) can then run it client-side, optionally on WebGPU. A parity
check compares ONNX Runtime logits to PyTorch on a fixed prompt.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from localagent.model import LocalAgentLM, ModelConfig


class _LogitsOnly(nn.Module):
    """ONNX-friendly wrapper: byte ids -> logits (drops the optional loss/cache outputs)."""

    def __init__(self, model: LocalAgentLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)[0]


class _LogitsAndHidden(nn.Module):
    """ONNX wrapper exposing BOTH outputs the browser bundle needs:

    * ``logits``  (B, T, vocab)   — next-byte distribution for decoding
    * ``hidden``  (B, T, d_model) — the post-final-norm features the tool/pointer heads read
      (== ``feats`` from ``LocalAgentLM.forward(..., return_hidden=True)``).
    """

    def __init__(self, model: LocalAgentLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor):
        logits, hidden = self.model(idx, return_hidden=True)
        return logits, hidden


def export(checkpoint: str, out_path: str, opset: int = 17, check: bool = True) -> None:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    model = LocalAgentLM(cfg).eval()
    model.load_state_dict(ck["state_dict"])
    wrap = _LogitsOnly(model).eval()

    dummy = torch.randint(0, cfg.vocab_size, (1, 16))
    torch.onnx.export(
        wrap, (dummy,), out_path, opset_version=opset, dynamo=False,
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}})
    print(f"wrote {out_path}")

    if check:
        import numpy as np
        import onnxruntime as ort
        with torch.no_grad():
            ref = wrap(dummy).numpy()
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        got = sess.run(["logits"], {"input_ids": dummy.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        print(f"parity vs PyTorch: max|Δ|={diff:.2e} ({'OK' if diff < 1e-3 else 'CHECK'})")
        # argmax agreement (what actually matters for decoding)
        agree = (ref.argmax(-1) == got.argmax(-1)).mean()
        print(f"argmax agreement: {agree*100:.1f}%")


# --------------------------------------------------------------------------------------------
# Browser bundle (onnxruntime-web: WASM + WebGPU)
# --------------------------------------------------------------------------------------------

def _round_nested(x, dp: int = 6):
    """Round a (possibly nested) python list of floats to ``dp`` decimals for JSON size."""
    if isinstance(x, list):
        return [_round_nested(v, dp) for v in x]
    return round(float(x), dp)


def _heads_json(ck: dict) -> dict:
    """Serialize tool_head + pointer_head weights as plain nested arrays (no onnx needed in JS).

    tool_head: a single Linear (d_model -> 22). JS computes  hidden[:, -1] @ W.T + b.
    pointer_head: per-arg copy mechanism. For arg a, with query q = arg_emb[a] (d_model):
        start_logit[t] = hidden[t] . (start_W @ q)   (i.e. hidden @ (start_W @ q))
        end_logit[t]   = hidden[t] . (end_W   @ q)
    We export arg_emb, start_W, end_W raw; JS reproduces the two matvecs + einsum.
    """
    from localagent.agent.pointer_head import ARG_IDX, PTR_ARGS
    from localagent.agent.tool_head import CLASSES

    th = ck["tool_head"]
    ph = ck["ptr_head"]
    tool_w = th["fc.weight"].cpu().tolist()   # (22, d_model)
    tool_b = th["fc.bias"].cpu().tolist()     # (22,)
    stop_index = CLASSES.index("text")

    return {
        "tool_head": {
            # weight[c] is the row for class c; logit_c = sum_d hidden[d]*weight[c][d] + bias[c]
            "weight": _round_nested(tool_w),          # shape (22, d_model)
            "bias": _round_nested(tool_b),            # shape (22,)
            "classes": list(CLASSES),                 # ordered, len 22
            "stop_index": stop_index,                 # index of "text" (no-tool / planner STOP)
        },
        "pointer_head": {
            # arg_emb[i] is the query vector for PTR_ARGS[i]; shape (n_args, d_model)
            "arg_emb": _round_nested(ph["arg_emb.weight"].cpu().tolist()),
            # start_W / end_W are (d_model, d_model). projected query qp = start_W @ arg_emb[i].
            # then start_logit[t] = hidden[t] . qp_start ;  end_logit[t] = hidden[t] . qp_end.
            # (matches PointerHead.logits: einsum('btd,bd->bt', feats, start(q)).)
            "start_W": _round_nested(ph["start.weight"].cpu().tolist()),  # (d_model, d_model)
            "end_W": _round_nested(ph["end.weight"].cpu().tolist()),      # (d_model, d_model)
            "args": list(PTR_ARGS),
            "arg_idx": dict(ARG_IDX),
            # decode rule JS must apply: start = argmax(start_logit);
            #   end_logit[:start] = -inf; end = argmax(end_logit); span is [start, end] inclusive.
        },
    }


def _meta_json(cfg: ModelConfig) -> dict:
    """Tokenization + toolset contract for the JS runtime (byte-accurate special markers)."""
    from localagent.agent.toolset import STANDARD_TOOLS
    from localagent.agent.tool_head import CLASSES
    from localagent.model import tokenizer as tk

    def byte_ids(s: str) -> list[int]:
        return list(s.encode("utf-8"))

    markers = {
        "user": {"text": tk.USER, "ids": byte_ids(tk.USER)},
        "assistant": {"text": tk.ASSISTANT, "ids": byte_ids(tk.ASSISTANT)},
        "tool": {"text": tk.TOOL, "ids": byte_ids(tk.TOOL)},
        "tool_call_open": {"text": tk.TOOL_CALL_OPEN, "ids": byte_ids(tk.TOOL_CALL_OPEN)},
        "tool_call_close": {"text": tk.TOOL_CALL_CLOSE, "ids": byte_ids(tk.TOOL_CALL_CLOSE)},
        "tool_response_open": {"text": tk.TOOL_RESPONSE_OPEN, "ids": byte_ids(tk.TOOL_RESPONSE_OPEN)},
        "tool_response_close": {
            "text": tk.TOOL_RESPONSE_CLOSE, "ids": byte_ids(tk.TOOL_RESPONSE_CLOSE)},
    }
    tools = []
    for t in STANDARD_TOOLS:
        props = t.parameters.get("properties", {})
        tools.append({
            "name": t.name,
            "description": t.description,
            "args": list(props.keys()),
            "schema": t.parameters,
        })
    return {
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "pad_id": tk.PAD_ID,
        "eos_id": tk.EOS_ID,
        "encoding": "utf-8-bytes",   # input_ids[i] = byte value (0..255); markers are literal UTF-8
        "markers": markers,
        "tools": tools,
        "tool_classes": list(CLASSES),
    }


def export_web(checkpoint: str, out_dir: str, fp16: bool = True, opset: int = 17,
               check: bool = True) -> dict:
    """Emit a browser-ready onnxruntime-web bundle (WASM + WebGPU) into ``out_dir``:

      * ``model.onnx``       — fp32, single graph, inputs ``input_ids`` (int64 [batch, seq]),
                               outputs ``logits`` ([batch, seq, vocab]) AND ``hidden``
                               ([batch, seq, d_model], the last hidden state the heads read).
      * ``model.fp16.onnx``  — fp16 variant (the web default), only when ``fp16=True``.
      * ``heads.json``       — tool_head + pointer_head weights as nested arrays (apply in JS).
      * ``meta.json``        — vocab/d_model/pad_id, byte-accurate markers, the standard toolset.

    Returns a dict of artifact paths + sizes + parity stats.
    """
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    model = LocalAgentLM(cfg).eval()
    model.load_state_dict(ck["state_dict"])
    wrap = _LogitsAndHidden(model).eval()

    fp32_path = os.path.join(out_dir, "model.onnx")
    dummy = torch.randint(0, cfg.vocab_size, (1, 16))
    torch.onnx.export(
        wrap, (dummy,), fp32_path, opset_version=opset, dynamo=False,
        input_names=["input_ids"], output_names=["logits", "hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            "hidden": {0: "batch", 1: "seq"},
        })
    print(f"wrote {fp32_path}")

    fp16_path = None
    if fp16:
        import onnx
        import onnxruntime as ort
        from onnxruntime.transformers.float16 import convert_float_to_float16
        fp16_path = os.path.join(out_dir, "model.fp16.onnx")
        m32 = onnx.load(fp32_path)
        # keep_io_types: input_ids stays int64; logits/hidden cast back to fp32 at the graph
        # boundary so JS reads ordinary Float32Array tensors regardless of internal fp16.
        m16 = convert_float_to_float16(m32, keep_io_types=True)
        # The raw fp16 graph collides with ORT's SimplifiedLayerNormFusion (a fp32 norm Constant
        # left dangling after cast insertion) and FAILS to load at ORT_ENABLE_ALL — which is the
        # onnxruntime-web default. Pre-optimize at EXTENDED offline and save the result: this
        # bakes the (working) RMSNorm fusion into the file, decomposes to plain opset-17 ops
        # (no com.microsoft kernels needed), and loads cleanly at ENABLE_ALL in WASM/WebGPU.
        raw16 = os.path.join(out_dir, "_raw.fp16.onnx")
        onnx.save(m16, raw16)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = fp16_path
        ort.InferenceSession(raw16, so, providers=["CPUExecutionProvider"])
        os.remove(raw16)
        print(f"wrote {fp16_path}")

    heads = _heads_json(ck)
    meta = _meta_json(cfg)
    heads_path = os.path.join(out_dir, "heads.json")
    meta_path = os.path.join(out_dir, "meta.json")
    with open(heads_path, "w") as f:
        json.dump(heads, f)
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"wrote {heads_path}  {meta_path}")

    stats = {
        "model.onnx": fp32_path,
        "model.fp16.onnx": fp16_path,
        "heads.json": heads_path,
        "meta.json": meta_path,
    }

    if check:
        import numpy as np
        import onnxruntime as ort
        with torch.no_grad():
            ref_logits, ref_hidden = wrap(dummy)
        ref_logits, ref_hidden = ref_logits.numpy(), ref_hidden.numpy()
        sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        got_logits, got_hidden = sess.run(["logits", "hidden"], {"input_ids": dummy.numpy()})
        dl = float(np.abs(ref_logits - got_logits).max())
        dh = float(np.abs(ref_hidden - got_hidden).max())
        print(f"fp32 parity: max|Δlogits|={dl:.2e}  max|Δhidden|={dh:.2e}")
        stats["fp32_logits_maxdiff"] = dl
        stats["fp32_hidden_maxdiff"] = dh
        if fp16_path is not None:
            s16 = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])
            g16_logits, _ = s16.run(["logits", "hidden"], {"input_ids": dummy.numpy()})
            stats["fp16_logits_drift"] = float(np.abs(ref_logits - g16_logits).max())
            print(f"fp16 vs fp32 logit drift: {stats['fp16_logits_drift']:.2e}")

    for name, path in list(stats.items()):
        if isinstance(path, str) and os.path.exists(path):
            stats[f"{name}_MB"] = os.path.getsize(path) / 1e6
    return stats

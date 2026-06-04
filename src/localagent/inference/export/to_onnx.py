"""Export the model to ONNX (Phase 9) — enables onnxruntime / onnxruntime-web (WASM + WebGPU).

Exports the full-sequence forward (logits over a byte prompt); browser runtimes
(onnxruntime-web, transformers.js) can then run it client-side, optionally on WebGPU. A parity
check compares ONNX Runtime logits to PyTorch on a fixed prompt.
"""

from __future__ import annotations

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

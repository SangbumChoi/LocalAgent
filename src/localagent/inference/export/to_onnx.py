"""Export to ONNX (Phase 9) — one graph, Execution Providers for CPU/GPU/NPU (incl. Ryzen AI)."""

from __future__ import annotations


def export(checkpoint: str, out_path: str) -> None:
    raise NotImplementedError("TODO(phase-9): torch.onnx export with dynamic axes + KV cache I/O")

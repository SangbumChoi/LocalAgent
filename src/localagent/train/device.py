"""One device abstraction so the *same* training/inference loop runs on GPU, CPU, or NPU-ish
backends (CUDA / MPS / XPU / CPU). Resolves device + an autocast dtype policy.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU/NPU
        return torch.device("xpu")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, spec: str = "auto") -> torch.dtype:
    if spec == "fp32":
        return torch.float32
    if spec == "fp16":
        return torch.float16
    if spec == "bf16":
        return torch.bfloat16
    # auto
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32  # CPU/MPS default to fp32 for stability


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type in ("cuda", "cpu", "xpu") and dtype != torch.float32:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()

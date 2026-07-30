"""One device abstraction so the *same* training/inference loop runs on GPU, CPU, or NPU-ish
backends (CUDA / MPS / XPU / CPU). Resolves device + an autocast dtype policy.
"""

from __future__ import annotations

import platform
from contextlib import nullcontext

import torch


_DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
}


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


def execution_metadata(
    *,
    requested_device: str,
    resolved_device: torch.device,
    requested_dtype: str,
    resolved_dtype: torch.dtype,
) -> dict[str, str | bool | int]:
    """Return the canonical JSON-safe runtime identity for one stage execution."""

    try:
        dtype_name = _DTYPE_NAMES[resolved_dtype]
    except KeyError as error:
        raise ValueError(f"unsupported resolved dtype {resolved_dtype}") from error
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "requested_dtype": str(requested_dtype),
        "resolved_dtype": dtype_name,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": bool(mps_backend and mps_backend.is_built()),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type in ("cuda", "cpu", "xpu") and dtype != torch.float32:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()

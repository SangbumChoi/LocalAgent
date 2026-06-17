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


class Amp:
    """Mixed-precision helper shared by every training loop (pretrain/SFT/GRPO).

    Wraps autocast + an (fp16-only) GradScaler behind one tiny interface so the loops stay
    backend-agnostic. It is a **no-op when disabled or in fp32**, so the CPU/fp32 path is
    byte-for-byte identical to the pre-AMP code:

        amp = Amp(device, dtype, enabled)
        with amp.autocast():            # mixed-precision forward (fp32 if disabled)
            loss = forward(...)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)              # call once per micro-batch (accumulation-safe)
        amp.step(opt, params)           # unscale -> clip -> step -> update, once per opt step

    bf16 needs no loss scaling; fp16 (older GPUs, e.g. T4) gets a real ``GradScaler``.
    """

    def __init__(self, device, dtype: torch.dtype, enabled: bool = True, max_norm: float = 1.0):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.max_norm = max_norm
        self.on = bool(enabled) and dtype != torch.float32 and self.device.type in ("cuda", "cpu", "xpu")
        # A scaler is only needed (and only valid) for fp16 on CUDA; disabled => transparent passthrough.
        use_scaler = self.on and self.device.type == "cuda" and dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def autocast(self):
        return autocast_ctx(self.device, self.dtype) if self.on else nullcontext()

    def backward(self, loss) -> None:
        self.scaler.scale(loss).backward()

    def step(self, opt, params) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, self.max_norm)
        self.scaler.step(opt)
        self.scaler.update()


def enable_tf32() -> None:
    """Allow TF32 matmuls/convs on Ampere+ GPUs — a free ~1.5-2x on fp32 paths. No-op elsewhere."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


"""Small screenshot-to-token bridge for optional multimodal experiments.

The bridge is deliberately independent of a pretrained vision tower: it is a compact patch
encoder that stays inside the model budget and can be trained from public screenshot/action rows.
It is disabled in all legacy configs unless ``vision_enabled`` is set explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.model.config import ModelConfig


class VisualPatchEncoder(nn.Module):
    """Encode ``[0, 1]`` RGB screenshots into a fixed visual-token prefix."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if not cfg.vision_enabled:
            raise ValueError("VisualPatchEncoder requires vision_enabled=True")
        self.image_size = cfg.vision_image_size
        self.patch = nn.Conv2d(
            3,
            cfg.vision_width,
            kernel_size=cfg.vision_patch_size,
            stride=cfg.vision_patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(cfg.vision_width, eps=cfg.norm_eps)
        self.proj = nn.Linear(cfg.vision_width, cfg.d_model, bias=False)
        self.position = nn.Parameter(torch.zeros(cfg.vision_tokens, cfg.d_model))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point tensors in [0, 1]")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")
        if images.numel() and (float(images.min()) < 0.0 or float(images.max()) > 1.0):
            raise ValueError("images must be in [0, 1]")
        resized = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        patches = self.patch(resized).flatten(2).transpose(1, 2)
        tokens = self.proj(self.norm(patches))
        return tokens + self.position.unsqueeze(0).to(tokens)

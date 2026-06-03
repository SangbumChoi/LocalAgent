"""Pretrain from scratch (Phase 2): next-token CE over packed shards.

AdamW/Muon, cosine LR w/ warmup, grad accumulation, bf16/fp16 autocast, checkpoint/resume.
Runs on GPU or CPU via train/device.py. Driven by configs/train/pretrain.yaml.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    raise NotImplementedError(
        "TODO(phase-2): data shards -> LocalAgentLM -> CE loss -> AdamW/cosine -> ckpt"
    )

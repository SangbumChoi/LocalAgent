"""Supervised fine-tune on agent Conversations (Phase 4).

Loss-masked to assistant + tool-call spans (don't train on user/tool-response tokens).
Function masking (Hammer): randomly rename/mask tools so the model generalizes across surface
forms instead of memorizing tool names. Driven by configs/train/sft.yaml.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    raise NotImplementedError(
        "TODO(phase-4): render Conversations to tokens with a loss mask + function masking; CE"
    )

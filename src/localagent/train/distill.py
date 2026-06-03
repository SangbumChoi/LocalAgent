"""Distillation (Phase 6).

modes:
  off_policy : sequence-KD — teacher generates trajectories, student imitates (default starter)
  on_policy  : reverse-KL — student samples, teacher scores its own rollouts (MiniLLM/OPD)

Trajectory-level: distill the teacher's whole tool-use rollout (reasoning + tool_call +
tool_response), not just the final answer (2505.17612), with optional first-thought prefix.
Driven by configs/train/distill.yaml.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    raise NotImplementedError(
        "TODO(phase-6): teacher backend + reverse_kl/seq_kd loss + on/off-policy sampling loop"
    )

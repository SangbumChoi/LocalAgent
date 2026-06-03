"""Synthetic agent-data generation (Phase 3).

Synthesis of tool-calling Conversations, combining the good parts of:
  - APIGen/xLAM : multi-stage verification (format -> execution -> semantic)
  - ToolACE     : multi-agent synthesis + complexity targeting + self-evolving API pool
  - Hammer      : irrelevance negatives + function masking

Driven by configs/data/agent_synth.yaml. The generator/verifier "teacher" is any
OpenAI-compatible endpoint or a local checkpoint.
"""

from __future__ import annotations

from localagent.data.schema import Conversation


def synthesize(config_path: str) -> None:
    """Generate a verified agent dataset to the configured JSONL path."""
    raise NotImplementedError(
        "TODO(phase-3): sample tools -> multi-agent dialog at target complexity -> "
        "inject irrelevance negatives -> dual-verify -> write Conversations"
    )


def sample_dialog(tools, complexity_bucket: str, generator) -> Conversation:
    raise NotImplementedError("TODO(phase-3): ToolACE-style multi-agent synthesis")


def verify(conv: Conversation, rule_based: bool, model_based: bool) -> bool:
    """Dual-layer verification: rule (schema/AST/executable) + model (semantic)."""
    raise NotImplementedError("TODO(phase-3): APIGen/ToolACE dual verification")

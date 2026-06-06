"""AST-based tool-call evaluation (Phase 5), BFCL-style.

Compare predicted vs reference calls as trees (name + normalized args), not strings. Splits:
single / parallel / multiple / multi-turn, plus an irrelevance (correct-abstention) score.
The `normalized()` form on ToolCall is the comparison key.
"""

from __future__ import annotations

from localagent.data.schema import ToolCall


def match_calls(pred: list[ToolCall], ref: list[ToolCall]) -> bool:
    """Order-insensitive exact match on (name, normalized-args). Used for parallel/multiple."""
    return sorted(c.normalized() for c in pred) == sorted(c.normalized() for c in ref)


def irrelevance_correct(pred: list[ToolCall]) -> bool:
    """For irrelevance samples, the model is correct iff it emitted NO tool call."""
    return len(pred) == 0


def score_dataset(jsonl_path: str, model) -> dict:
    """Run the model over a tool-eval JSONL and return per-split accuracy."""
    raise NotImplementedError("TODO(phase-5): drive model, parse calls, aggregate AST metrics")

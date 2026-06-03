"""Eval harness (Phase 5): run a suite, write a JSON scorecard.

Also drives a multi-turn simulated-user + tool-sandbox eval (tau-bench-lite) and the
export parity check (compare an exported runtime vs the PyTorch reference on fixed prompts).
"""

from __future__ import annotations


def run(checkpoint: str, suite: str = "all", out: str = "runs/eval/report.json") -> dict:
    raise NotImplementedError(
        "TODO(phase-5): load ckpt -> tool_eval + text_eval + multi-turn sandbox -> JSON report"
    )


def parity_check(reference_model, exported_path: str, prompts: list[str]) -> dict:
    raise NotImplementedError("TODO(phase-9): compare exported runtime vs PyTorch reference")

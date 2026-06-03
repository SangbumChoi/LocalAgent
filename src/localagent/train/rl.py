"""Optional RL stage (Phase 10): GRPO with a tool-correctness + task-success reward.

Group-relative policy optimization over agent rollouts; reward = AST-correct tool calls +
end-to-end task success from the multi-turn eval sandbox.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    raise NotImplementedError("TODO(phase-10): GRPO rollouts + tool/task reward")

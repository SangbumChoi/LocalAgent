"""Pipeline orchestration (Phase 0 stub -> grows with phases).

Wires stages into a runnable DAG with artifact tracking:

    pretrain -> sft -> distill -> eval -> export
                          ^                  |
                          |   data flywheel  |
                          +----- serve <-----+

`scripts/speedrun.sh` is the one-command, toy-scale, end-to-end entry point.
"""

from __future__ import annotations

# Stage name -> (module, callable). Kept declarative so the CLI/speedrun can run any sub-DAG.
STAGES = {
    "pretrain": ("localagent.train.pretrain", "run"),
    "sft": ("localagent.train.sft", "run"),
    "distill": ("localagent.train.distill", "run"),
    "rl": ("localagent.train.rl", "run"),
    "eval": ("localagent.eval.harness", "run"),
}


def run_stage(stage: str, config_path: str) -> None:
    import importlib

    if stage not in STAGES:
        raise SystemExit(f"unknown stage '{stage}'. known: {', '.join(STAGES)}")
    module_name, fn_name = STAGES[stage]
    fn = getattr(importlib.import_module(module_name), fn_name)
    fn(config_path)

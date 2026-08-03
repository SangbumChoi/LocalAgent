#!/usr/bin/env python3
"""Run a bounded AppWorld evaluation of a LocalAgent checkpoint.

AppWorld is an optional external dependency.  The selected tasks must be from its train/dev
splits, where ground-truth verifiers are available in the public data bundle.  This adapter
deliberately does not translate LocalAgent's compact tool vocabulary into AppWorld Python/API
calls: a zero-action result is therefore a native checkpoint baseline and an explicit interface
gap, not a claimed AppWorld agent score.  The receipt keeps task text out of committed artifacts.

Example (with AppWorld installed in an isolated environment)::

    APPWORLD_ROOT=/tmp/appworld-data \
      PYTHONPATH=/tmp/appworld-venv/lib/python3.12/site-packages:src \
      python scripts/evaluate_appworld_checkpoint.py \
      --checkpoint runs/sft-mind2web-trajectory-continuation-20260802/latest.pt \
      --task 29caf6f_1 --task 771d8fc_1 --task 530b157_1 \
      --report /tmp/appworld-current-checkpoint.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any

_TOOL_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\(")


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _text_hash(text: str) -> dict[str, int | str]:
    encoded = text.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _task_ids(root: Path, split: str) -> set[str]:
    path = root / "data" / "datasets" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"AppWorld split file not found: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _registry():
    from localagent.agent.tools import ToolRegistry
    from localagent.agent.toolset import STANDARD_TOOLS

    registry = ToolRegistry()
    for tool in STANDARD_TOOLS:
        registry.register(
            tool,
            (lambda name: (lambda **kwargs: {"ok": name, "args": kwargs}))(tool.name),
        )
    return registry


def _tracker_summary(tracker: Any) -> dict[str, Any]:
    payload = tracker.to_dict()
    passes = payload.get("passes", [])
    failures = payload.get("failures", [])
    return {
        "success": bool(payload.get("success", False)),
        "difficulty": payload.get("difficulty"),
        "num_tests": int(payload.get("num_tests", len(passes) + len(failures))),
        "passed": len(passes),
        "failed": len(failures),
        "pass_labels": sorted(str(item.get("label", "")) for item in passes),
        "failure_labels": sorted(str(item.get("label", "")) for item in failures),
    }


def _verify_runner_contract(*, AppWorld: Any, task_id: str, experiment_name: str) -> dict[str, Any]:
    """Run one bundled ground-truth solution to prove the native verifier is live."""

    with AppWorld(
        task_id=task_id,
        experiment_name=f"{experiment_name}_oracle",
        load_ground_truth=True,
        ground_truth_mode="full",
        max_interactions=256,
    ) as world:
        ground_truth = world.task.ground_truth
        if ground_truth is None:
            raise RuntimeError(f"AppWorld ground truth missing for contract task {task_id!r}")
        code = ground_truth.compiled_solution_code + "\nsolution(apis, requester)"
        world.execute(code)
        tracker = world.evaluate()
    summary = _tracker_summary(tracker)
    summary["task_id"] = task_id
    return summary


def evaluate(
    *, checkpoint: Path, root: Path, task_ids: list[str], report: Path, experiment_name: str
) -> dict[str, Any]:
    try:
        appworld_version = importlib.metadata.version("appworld")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "AppWorld is optional; install it in an isolated environment before running this "
            "script (see docs/REALISTIC_AGENT_RESEARCH.md)."
        ) from error

    from appworld import AppWorld, update_root
    from localagent.agent.runtime import Agent

    root = root.resolve()
    resolved_root = Path(update_root(str(root))).resolve()
    if resolved_root != root:
        raise RuntimeError(
            f"APPWORLD_ROOT resolved to {str(resolved_root)!r}, expected {str(root)!r}; set it before "
            "importing AppWorld."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not task_ids:
        raise ValueError("at least one --task is required")
    for task_id in task_ids:
        if not (root / "data" / "tasks" / task_id / "specs.json").is_file():
            raise FileNotFoundError(f"AppWorld task specs not found for {task_id!r}")

    agent = Agent.from_checkpoint(checkpoint, _registry())
    contract_verification = _verify_runner_contract(
        AppWorld=AppWorld, task_id=task_ids[0], experiment_name=experiment_name
    )
    task_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_spec_path = root / "data" / "tasks" / task_id / "specs.json"
        task_spec = json.loads(task_spec_path.read_text(encoding="utf-8"))
        instruction = str(task_spec["instruction"])
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            ground_truth_mode="full",
            max_interactions=1,
        ) as world:
            output = agent.chat(instruction)
            tracker = world.evaluate()
            tracker_summary = _tracker_summary(tracker)
        match = _TOOL_RE.search(output)
        task_records.append(
            {
                "task_id": task_id,
                "task_spec": _sha256(task_spec_path),
                "instruction": _text_hash(instruction),
                "model_output": _text_hash(output),
                "predicted_tool": match.group(1) if match else None,
                "action_replayed": False,
                "native_api_calls": 0,
                "evaluation": tracker_summary,
            }
        )

    result = {
        "kind": "localagent_appworld_checkpoint_native_probe",
        "schema_version": 1,
        "runner": {
            "package": "appworld",
            "version": appworld_version,
            "root": str(root),
            "data_version": (root / "data" / "version.txt").read_text(encoding="utf-8").strip(),
            "split_policy": "caller-selected train/dev tasks with full public ground-truth verifiers",
            "contract_verification": {
                "tasks": 1,
                "passed": int(contract_verification["success"]),
                "result": contract_verification,
            },
        },
        "checkpoint": _sha256(checkpoint),
        "configuration": {
            "experiment_name": experiment_name,
            "tasks": task_ids,
            "action_translation": "disabled",
            "max_interactions": 1,
        },
        "environment": {
            "native_runtime_executed": True,
            "environment_reset_per_task": True,
            "external_accounts": False,
            "screenshots": False,
            "state_side_effects": "isolated AppWorld task databases only",
        },
        "tasks": task_records,
        "summary": {
            "tasks": len(task_records),
            "native_successes": sum(int(item["evaluation"]["success"]) for item in task_records),
            "native_success_rate": sum(int(item["evaluation"]["success"]) for item in task_records)
            / len(task_records),
            "action_replayed": 0,
            "native_api_calls": 0,
        },
        "claim_boundary": (
            "Native AppWorld reset/evaluation of the current LocalAgent checkpoint only. The model "
            "emits LocalAgent tool syntax, while AppWorld expects Python/API actions; no action was "
            "translated or replayed. The score is a zero-action interface baseline, not an AppWorld "
            "leaderboard result, AppWorld-UL result, or evidence of email/SMS/Spotify task success."
        ),
    }
    result["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("APPWORLD_ROOT", ".")))
    parser.add_argument("--task", dest="tasks", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-name", default="localagent_appworld_native_probe")
    args = parser.parse_args()
    result = evaluate(
        checkpoint=args.checkpoint,
        root=args.root,
        task_ids=args.tasks,
        report=args.report,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

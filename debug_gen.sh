#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
timeout 300 .venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "scripts")
from eval_suite import LocalAgentAdapter, build_tasks, PUBLIC, parse_call
tasks = build_tasks(PUBLIC / "toolace-eval.jsonl", 3)
for tag in ("runs/region/data-union/model.pt", "runs/region/data-synthetic/model.pt"):
    adapter = LocalAgentAdapter(tag, "cuda")
    print("=====", tag)
    for task in tasks[:2]:
        raw = adapter.predict(task, 64)
        print("  gold:", task.gold_name, task.gold_arguments)
        print("  raw :", repr(raw[:200]))
        print("  parsed:", parse_call(raw))
PY

#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/majority_baseline.py > explog/majority.log 2>&1' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

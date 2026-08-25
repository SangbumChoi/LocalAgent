#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/tool_overlap.py > explog/overlap.log 2>&1' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

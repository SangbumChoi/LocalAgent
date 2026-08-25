#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/probe_lfm2.py > explog/probe_lfm2.log 2>&1
echo "probe-lfm2 rc=$?" >> explog/EVALSUITE_STATUS.txt' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

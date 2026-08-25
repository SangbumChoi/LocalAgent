#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
timeout 900 .venv/bin/python scripts/eval_suite.py --model localagent:runs/region/data-union/model.pt \
  --out /tmp/suite_local.json --rows 25 2>&1 | tail -6

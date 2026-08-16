#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
timeout 600 .venv/bin/python scripts/eval_suite.py --model dispatch:retrieve+ground \
  --out /tmp/suite_dispatch.json --rows 50 --device cpu 2>&1 | tail -6

#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
ls -la runs/sft-catalog-10m/ 2>/dev/null | head -4
timeout 1800 .venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-catalog-10m/latest.pt \
  --rows 60 --device cuda --out /tmp/catalog10.json 2>&1 | tail -8

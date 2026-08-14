#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
timeout 1800 .venv/bin/python scripts/build_merged_v2.py --cap 6000 --synthetic-episodes 1200 \
  --out-dir data/merged-v2 2>&1 | tail -25

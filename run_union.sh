#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
timeout 900 .venv/bin/python scripts/build_union_dataset.py --cap 1000 --episodes 360 \
  --out data/public/localagent-union-v1.jsonl \
  --manifest data/public/localagent-union-v1.manifest.json 2>&1 | tail -15
ls -lh data/public/localagent-union-v1.jsonl 2>/dev/null

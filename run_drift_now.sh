#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
for arm in data-union region-embed region-attn; do
  [ -f "runs/region/$arm/model.pt" ] || continue
  echo "=== $arm ==="
  timeout 600 .venv/bin/python scripts/region_drift.py --checkpoint "runs/region/$arm/model.pt" \
    --out "runs/region/$arm/drift.json" 2>&1 | tail -8
done

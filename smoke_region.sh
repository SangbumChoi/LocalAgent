#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
timeout 600 .venv/bin/python scripts/region_transfer.py --region full --data union \
  --out runs/region/_smoke --steps 12 --n-synth 200 --n-public 200 2>&1 | tail -20
echo "SMOKE_RC=$?"

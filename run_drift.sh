#!/usr/bin/env bash
# Per-region drift between the public donor and each trained arm, for the mechanism section.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/REGION_STATUS.txt
while ! grep -q WAVE4_DONE "$STATUS" 2>/dev/null; do sleep 60; done
for arm in data-union data-synthetic region-attn region-ffn region-scratch; do
  [ -f "runs/region/$arm/model.pt" ] || continue
  .venv/bin/python scripts/region_drift.py --checkpoint "runs/region/$arm/model.pt" \
    --out "runs/region/$arm/drift.json" >> explog/region_drift.log 2>&1
done
echo DRIFT_ALL_DONE >> "$STATUS"

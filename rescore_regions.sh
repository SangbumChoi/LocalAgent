#!/usr/bin/env bash
# The first two region arms were scored before the parser accepted the Pythonic call form; rescore
# them so the region table is one parser version throughout.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
for _ in $(seq 1 90); do
  grep -q REGION_CATALOG_DONE explog/REGION_CATALOG_STATUS.txt 2>/dev/null && break
  sleep 60
done
for region in attn ffn; do
  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-rc-$region/latest.pt" \
    --rows 200 --device cuda --out "runs/evalsuite/rc-$region.json" \
    > "explog/rc_eval_${region}_v2.log" 2>&1
  echo "$region rescored rc=$? [parser-v2]" >> explog/REGION_CATALOG_STATUS.txt
done
echo REGION_RESCORE_DONE >> explog/REGION_CATALOG_STATUS.txt

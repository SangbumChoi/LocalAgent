#!/usr/bin/env bash
# ToolSandbox first-call suite across every model with an existing artifact.
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src
ev() { .venv/bin/python scripts/eval_suite.py --model "$1" --rows 200 --device cuda --suites toolsandbox,mcpatlas,mobileactions --out "runs/evalsuite/ts-$2.json" >> explog/ts_sweep.log 2>&1 && echo "ts $2 done" >> explog/TS_STATUS.txt; }
for arm in sft-arm2-96m sft-arm3-96m sft-fresh-96m sft-big-96m; do
  [ -f runs/$arm/latest.pt ] && ev "catalog:runs/$arm/latest.pt" "$arm"
done
for d in data/baselines/*/; do
  name=$(basename "$d")
  [ "$name" = gemma-3-270m-it ] && continue
  [ -f "$d/config.json" ] && ev "hf:$d" "rel-$name"
  [ -d "runs/lora/$name" ] && ev "lora:$d|runs/lora/$name" "ft-$name"
done
echo TS-SWEEP-DONE >> explog/TS_STATUS.txt

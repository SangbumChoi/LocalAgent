#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
grep -q BASELINES_DONE explog/baselines.log 2>/dev/null || { echo "baselines still downloading"; tail -1 explog/baselines.log; exit 0; }
timeout 900 .venv/bin/python scripts/eval_suite.py --model localagent:runs/region/data-union/model.pt \
  --out /tmp/suite_local.json --rows 25 2>&1 | tail -6
timeout 1200 .venv/bin/python scripts/eval_suite.py --model hf:data/baselines/SmolLM2-135M-Instruct \
  --out /tmp/suite_smol.json --rows 25 2>&1 | tail -6

#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/normalize_bfcl.py --per-category 150 > explog/bfcl3_norm.log 2>&1
echo "norm rc=$?" >> explog/BFCL3_STATUS.txt
.venv/bin/python scripts/prompt_fit.py > explog/bfcl3_fit.log 2>&1
echo "fit rc=$?" >> explog/BFCL3_STATUS.txt
.venv/bin/python scripts/chance_baseline.py > explog/chance8.log 2>&1
echo BFCL3_DONE >> explog/BFCL3_STATUS.txt' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

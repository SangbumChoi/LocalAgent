#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
: > explog/TOOLBENCH_STATUS.txt
.venv/bin/python scripts/normalize_toolbench.py --limit 200 > explog/toolbench_normalize.log 2>&1
echo "normalize rc=$?" >> explog/TOOLBENCH_STATUS.txt
.venv/bin/python scripts/chance_baseline.py --out runs/evalsuite/chance-baseline.json \
  >> explog/toolbench_normalize.log 2>&1
echo "chance rc=$?" >> explog/TOOLBENCH_STATUS.txt
.venv/bin/python scripts/score_toolbench.py --kinds catalog --rows 200 --device cuda \
  > explog/toolbench_score.log 2>&1
echo CATALOG_ARMS_DONE >> explog/TOOLBENCH_STATUS.txt' </dev/null >/dev/null 2>&1 &
sleep 3; echo TB_FACE_LAUNCHED

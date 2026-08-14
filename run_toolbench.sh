#!/usr/bin/env bash
# Normalize ToolBench, then add it to every existing report by replaying each arm's own spec.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/TOOLBENCH_STATUS.txt

.venv/bin/python scripts/normalize_toolbench.py --limit 200 > explog/toolbench_normalize.log 2>&1
echo "normalize rc=$?" >> "$STATUS"
.venv/bin/python scripts/chance_baseline.py --out runs/evalsuite/chance-baseline.json \
  >> explog/toolbench_normalize.log 2>&1
echo "chance rc=$?" >> "$STATUS"
.venv/bin/python scripts/score_toolbench.py --suite toolbench --rows 200 --device cuda \
  > explog/toolbench_score.log 2>&1

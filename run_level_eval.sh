#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/eval_by_level.py --ckpt runs/ablate/cuda-s0-x10/model.pt \
  --out runs/eval_by_level_x10.json --device cuda > explog/eval_by_level.log 2>&1
echo LEVEL_RC=$? >> explog/eval_by_level.log' </dev/null >/dev/null 2>&1 &
sleep 3; echo launched_level_eval

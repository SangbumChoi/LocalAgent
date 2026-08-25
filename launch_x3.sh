#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONUNBUFFERED=1 PYTHONPATH=src
.venv/bin/python scripts/ablate_flywheel.py --rounds 5 --out runs/ablate/cuda-s0-x3 --device cuda --seed 0 --sft-scale 3 > explog/ablate_cuda-s0-x3.log 2>&1
echo "ABL cuda-s0-x3 rc=$?" >> explog/ABLATE_STATUS.txt' </dev/null >/dev/null 2>&1 &
sleep 2; pgrep -cf ablate_flywheel; echo relaunched_x3

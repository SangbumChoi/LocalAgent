#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash run_eval_grid.sh </dev/null >/dev/null 2>&1 &
sleep 3; echo eval_grid_launched

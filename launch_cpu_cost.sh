#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash run_cpu_cost.sh </dev/null >/dev/null 2>&1 &
sleep 2; echo cpu_cost_armed

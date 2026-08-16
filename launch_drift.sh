#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash run_drift.sh </dev/null >/dev/null 2>&1 &
sleep 2; echo drift_armed

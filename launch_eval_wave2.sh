#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash run_eval_wave2.sh </dev/null >/dev/null 2>&1 &
sleep 2; echo eval_wave2_armed

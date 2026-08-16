#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash run_wave4.sh </dev/null >/dev/null 2>&1 &
sleep 2; echo wave4_armed

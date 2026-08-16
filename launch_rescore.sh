#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash rerun_localagent_evals.sh </dev/null >/dev/null 2>&1 &
sleep 3; echo rescore_launched

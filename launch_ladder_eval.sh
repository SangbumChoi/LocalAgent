#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash eval_ladder_chain.sh </dev/null >/dev/null 2>&1 &
sleep 2; echo ladder_eval_armed

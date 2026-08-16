#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash run_seed_spread.sh 101 202 303 </dev/null >/dev/null 2>&1 &
echo LAUNCHED $!

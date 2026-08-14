#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash run_size_ratio.sh </dev/null >/dev/null 2>&1 &
echo LAUNCHED

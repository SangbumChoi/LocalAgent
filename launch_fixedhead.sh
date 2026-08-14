#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash score_fixedhead.sh </dev/null >/dev/null 2>&1 &
echo LAUNCHED

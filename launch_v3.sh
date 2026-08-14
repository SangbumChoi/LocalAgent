#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash score_all_v3.sh ours   </dev/null >/dev/null 2>&1 &
setsid nohup bash score_all_v3.sh public </dev/null >/dev/null 2>&1 &
echo LAUNCHED_BOTH

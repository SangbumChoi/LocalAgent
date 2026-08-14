#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
for _ in $(seq 1 60); do grep -q PROFILED_DONE explog/PROFILED_STATUS.txt 2>/dev/null && break; sleep 30; done
setsid nohup bash score_v6.sh new     </dev/null >/dev/null 2>&1 &
setsid nohup bash score_v6.sh rescore </dev/null >/dev/null 2>&1 &
echo LAUNCHED_BOTH

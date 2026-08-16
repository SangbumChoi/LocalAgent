#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash rescore_all.sh </dev/null > explog/rescore_all_driver.log 2>&1 &
echo LAUNCHED $!

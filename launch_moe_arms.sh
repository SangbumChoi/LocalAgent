#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash run_moe_arms.sh </dev/null >/dev/null 2>&1 &
sleep 3; ls -lh runs/moe-donor-frozen.pt 2>/dev/null; echo moe_arms_armed

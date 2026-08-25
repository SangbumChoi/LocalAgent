#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/localagent train pretrain configs/train/pretrain-moe-donor.yaml > explog/moe_donor.log 2>&1
echo "MOE_DONOR rc=$?" >> explog/REGION_STATUS.txt' </dev/null >/dev/null 2>&1 &
sleep 3; echo moe_donor_launched

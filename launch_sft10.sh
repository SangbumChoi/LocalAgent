#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/localagent train sft configs/train/sft-catalog-10m.yaml > explog/sft_catalog_10m.log 2>&1
echo "sft-catalog-10m rc=$?" >> explog/LADDER_STATUS.txt' </dev/null >/dev/null 2>&1 &
sleep 20; tail -5 explog/sft_catalog_10m.log 2>/dev/null; echo launched

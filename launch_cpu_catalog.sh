#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-catalog-10m/latest.pt --rows 60 \
  --device cpu --out runs/evalsuite-cpu/catalog-10m.json > explog/cpu_catalog_10m.log 2>&1
echo "cpu-catalog-10m rc=$?" >> explog/EVALSUITE_STATUS.txt' </dev/null >/dev/null 2>&1 &
echo LAUNCHED_CPU $!

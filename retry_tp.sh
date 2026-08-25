#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python scripts/throughput.py --model hf:data/baselines/SmolLM2-360M-Instruct \
  --out runs/throughput/smollm2-360m.json --threads 4 > explog/tp_smollm2-360m.log 2>&1
echo "smollm2-360m retry rc=$?" >> explog/THROUGHPUT_STATUS.txt' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for pair in LFM2.5-230M:lfm25-230m granite-4.0-h-350m:granite-h-350m granite-4.0-350m:granite-350m; do
  .venv/bin/python scripts/throughput.py --model "hf:data/baselines/${pair%%:*}" \
    --out "runs/throughput/${pair#*:}.json" --threads 4 > "explog/tpn_${pair#*:}.log" 2>&1
  echo "${pair#*:} rc=$?" >> explog/TPNEW_STATUS.txt
done
echo TPNEW_DONE >> explog/TPNEW_STATUS.txt

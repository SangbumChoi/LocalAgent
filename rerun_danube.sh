#!/usr/bin/env bash
# danube3 rejects a system turn in its chat template; re-run it once the fix is in place.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
for _ in $(seq 1 200); do
  grep -q PUBLIC_V3_DONE explog/V3_STATUS.txt 2>/dev/null && break
  sleep 30
done
.venv/bin/python scripts/eval_suite.py --model hf:data/baselines/h2o-danube3-500m-chat \
  --rows 200 --device cuda --out runs/evalsuite/danube3-500m.json \
  > explog/v3_danube3-500m.log 2>&1
echo "danube3-500m rc=$? [system-folded]" >> explog/V3_STATUS.txt
OMP_NUM_THREADS=4 .venv/bin/python scripts/eval_suite.py \
  --model hf:data/baselines/h2o-danube3-500m-chat --rows 40 --device cpu \
  --out runs/evalsuite-cpu/danube3-500m.json > explog/cpu_danube3-500m.log 2>&1
echo "cpu-danube3-500m rc=$?" >> explog/CPU_STATUS.txt
echo DANUBE_DONE >> explog/V3_STATUS.txt

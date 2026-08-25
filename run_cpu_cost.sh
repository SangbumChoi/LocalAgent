#!/usr/bin/env bash
# Same tasks, same harness, CPU only: the cost axis of the dashboard's Pareto view.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
STATUS=explog/EVALSUITE_STATUS.txt
while ! grep -q EVALSUITE_DONE "$STATUS" 2>/dev/null; do sleep 60; done
mkdir -p runs/evalsuite-cpu
for entry in "dispatch dispatch:retrieve+ground" \
             "smollm2-135m hf:data/baselines/SmolLM2-135M-Instruct" \
             "smollm2-360m hf:data/baselines/SmolLM2-360M-Instruct" \
             "qwen25-05b hf:data/baselines/Qwen2.5-0.5B-Instruct" \
             "localagent-union localagent:runs/region/data-union/model.pt"; do
  set -- $entry
  .venv/bin/python scripts/eval_suite.py --model "$2" --rows 12 --device cpu \
    --out "runs/evalsuite-cpu/$1.json" > "explog/cpucost_$1.log" 2>&1
  echo "cpu-$1 rc=$?" >> "$STATUS"
done
echo CPU_COST_DONE >> "$STATUS"

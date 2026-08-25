#!/usr/bin/env bash
# Tokens per second on four CPU threads, the axis the deployment target is actually chosen on.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
STATUS=explog/THROUGHPUT_STATUS.txt
one() {
  .venv/bin/python scripts/throughput.py --model "$2" --out "runs/throughput/$1.json" \
    --threads 4 > "explog/tp_$1.log" 2>&1
  echo "$1 rc=$?" >> "$STATUS"
}
for size in 10m 16m 35m 96m; do one "catalog-$size" "catalog:runs/sft-catalog-$size/latest.pt"; done
for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2-350M:lfm2-350m \
            SmolLM2-360M-Instruct:smollm2-360m h2o-danube3-500m-chat:danube3-500m \
            Qwen2.5-0.5B-Instruct:qwen25-05b Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b \
            LFM2-700M:lfm2-700m Qwen3-0.6B:qwen3-06b; do
  one "${pair#*:}" "hf:data/baselines/${pair%%:*}"
done
echo THROUGHPUT_ALL_DONE >> "$STATUS"

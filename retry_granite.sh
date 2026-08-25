#!/usr/bin/env bash
# Granite needs two accommodations: the hybrid variant materialises a very large attention tensor
# during training (smaller batch), and this transformers version raises from its KV cache at
# generation (handled by a cacheless retry in the harness).
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STATUS=explog/GRANITE_STATUS.txt
for _ in $(seq 1 90); do
  grep -q NEW_MODELS_DONE explog/V6_STATUS.txt 2>/dev/null && break
  grep -q INVSEED_DONE explog/INVSEED_STATUS.txt 2>/dev/null && break
  sleep 60
done
score() { .venv/bin/python scripts/eval_suite.py --model "$2" --rows 200 --device cuda \
    --out "runs/evalsuite/$1.json" > "explog/g_$1.log" 2>&1; echo "$1 rc=$?" >> "$STATUS"; }

score granite-350m hf:data/baselines/granite-4.0-350m
for pair in granite-4.0-h-350m:granite-h-350m granite-4.0-350m:granite-350m; do
  base="data/baselines/${pair%%:*}"; tag="${pair#*:}"
  .venv/bin/python scripts/finetune_public.py --base "$base" --out "runs/lora/$tag" \
    --steps 600 --batch-size 2 --max-length 768 > "explog/g_ft_$tag.log" 2>&1
  echo "ft-$tag rc=$?" >> "$STATUS"
  score "ft-$tag" "lora:$base|runs/lora/$tag"
done
OMP_NUM_THREADS=4 .venv/bin/python scripts/throughput.py --model hf:data/baselines/granite-4.0-350m \
  --out runs/throughput/granite-350m.json --threads 4 > explog/g_tp.log 2>&1
echo "tp rc=$?" >> "$STATUS"
echo GRANITE_DONE >> "$STATUS"

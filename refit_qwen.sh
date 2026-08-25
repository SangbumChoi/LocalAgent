#!/usr/bin/env bash
# The Qwen2.5 pair only: their adapters went NaN through an inf gradient that a loss-only guard
# could not see. Everything else in the block trained with zero skipped steps and is untouched.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/FT3_STATUS.txt
for pair in Qwen2.5-0.5B-Instruct:qwen25-05b Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b; do
  base="data/baselines/${pair%%:*}"; tag="${pair#*:}"; start=$SECONDS
  .venv/bin/python scripts/finetune_public.py --base "$base" --out "runs/lora/$tag" \
    --steps 600 > "explog/ft3_$tag.log" 2>&1
  echo "ft-$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
  .venv/bin/python scripts/eval_suite.py --model "lora:$base|runs/lora/$tag" --rows 200 \
    --device cuda --out "runs/evalsuite/ft-$tag.json" > "explog/eval_ft3_$tag.log" 2>&1
  echo "eval-ft-$tag rc=$?" >> "$STATUS"
done
.venv/bin/python scripts/androidcontrol_clean.py \
  --model "lora:data/baselines/Qwen2.5-0.5B-Instruct|runs/lora/qwen25-05b" --device cuda \
  --out runs/evalsuite-clean/ft-qwen25-05b.json > explog/clean_ft3_qwen.log 2>&1
echo "clean rc=$?" >> "$STATUS"
echo FT3_DONE >> "$STATUS"

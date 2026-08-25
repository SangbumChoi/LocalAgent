#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
S=explog/CLEAN_STATUS.txt
one() { .venv/bin/python scripts/androidcontrol_clean.py --model "$2" --device cuda \
          --out "runs/evalsuite-clean/$1.json" > "explog/clean_$1.log" 2>&1
        echo "$1 rc=$?" >> "$S"; }
for size in 10m 16m 96m; do one "catalog-$size" "catalog:runs/sft-catalog-$size/latest.pt"; done
for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2-350M:lfm2-350m \
            Qwen2.5-0.5B-Instruct:qwen25-05b; do
  base="data/baselines/${pair%%:*}"; tag="${pair#*:}"
  one "$tag" "hf:$base"
  [ -d "runs/lora/$tag" ] && one "ft-$tag" "lora:$base|runs/lora/$tag"
done
echo CLEAN_ALL_DONE >> "$S"

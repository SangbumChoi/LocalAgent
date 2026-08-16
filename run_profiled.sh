#!/usr/bin/env bash
# Same midtrain parent, same corpus, same schedule — only the per-module learning rates differ,
# and they come from the open models' measured adaptation profile.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/PROFILED_STATUS.txt
sed -e "s|out_dir: runs/sft-distill2-10m|out_dir: runs/sft-profiled-10m|" \
    configs/train/sft-distill2-10m.yaml > configs/train/sft-profiled-10m.yaml
.venv/bin/python scripts/profiled_sft.py --profile runs/analysis/lora_profile.json \
  --config configs/train/sft-profiled-10m.yaml > explog/profiled_sft.log 2>&1
echo "sft rc=$?" >> "$STATUS"
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-profiled-10m/latest.pt \
  --rows 200 --device cuda --out runs/evalsuite/profiled-10m.json \
  > explog/profiled_eval.log 2>&1
echo "eval rc=$?" >> "$STATUS"
echo PROFILED_DONE >> "$STATUS"

#!/usr/bin/env bash
# Same intervention with the profile inverted: does the measured adaptation shape carry signal at
# all, and if so in which direction?
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/PROFILED_STATUS.txt
sed -e "s|out_dir: runs/sft-distill2-10m|out_dir: runs/sft-profinv-10m|" \
    configs/train/sft-distill2-10m.yaml > configs/train/sft-profinv-10m.yaml
.venv/bin/python scripts/profiled_sft.py --profile runs/analysis/lora_profile.json \
  --config configs/train/sft-profinv-10m.yaml --invert > explog/profinv_sft.log 2>&1
echo "inv sft rc=$?" >> "$STATUS"
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-profinv-10m/latest.pt \
  --rows 200 --device cuda --out runs/evalsuite/profinv-10m.json > explog/profinv_eval.log 2>&1
echo "inv eval rc=$?" >> "$STATUS"
echo PROFINV_DONE >> "$STATUS"

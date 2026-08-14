#!/usr/bin/env bash
# Widen the student's catalog exposure, then run the identical chain so the only change is data.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/WIDE_STATUS.txt
.venv/bin/python scripts/build_wide_corpus.py --out data/wide/train.jsonl \
  > explog/wide_corpus.log 2>&1
echo "corpus rc=$?" >> "$STATUS"

sed -e "s|path: data/merged-v2/train.jsonl|path: data/wide/train.jsonl|" \
    -e "s|out_dir: runs/midtrain-catalog-10m|out_dir: runs/midtrain-wide-10m|" \
    configs/train/midtrain-catalog-10m.yaml > configs/train/midtrain-wide-10m.yaml
sed -e "s|path: data/merged-v2/train.jsonl|path: data/wide/train.jsonl|" \
    -e "s|init_from: runs/midtrain-catalog-10m/latest.pt|init_from: runs/midtrain-wide-10m/latest.pt|" \
    -e "s|out_dir: runs/sft-catalog-10m|out_dir: runs/sft-wide-10m|" \
    configs/train/sft-catalog-10m.yaml > configs/train/sft-wide-10m.yaml

.venv/bin/localagent train midtrain configs/train/midtrain-wide-10m.yaml \
  > explog/wide_midtrain.log 2>&1
echo "midtrain rc=$?" >> "$STATUS"
.venv/bin/localagent train sft configs/train/sft-wide-10m.yaml > explog/wide_sft.log 2>&1
echo "sft rc=$?" >> "$STATUS"
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-wide-10m/latest.pt \
  --rows 200 --device cuda --out runs/evalsuite/wide-10m.json > explog/wide_eval.log 2>&1
echo "eval rc=$?" >> "$STATUS"
.venv/bin/python scripts/androidcontrol_clean.py --model catalog:runs/sft-wide-10m/latest.pt \
  --device cuda --out runs/evalsuite-clean/wide-10m.json > explog/wide_clean.log 2>&1
echo "clean rc=$?" >> "$STATUS"
echo WIDE_DONE >> "$STATUS"

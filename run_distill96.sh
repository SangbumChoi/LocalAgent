#!/usr/bin/env bash
# The winning data recipe at the largest rung: can the architecture clear chance-in-catalog on
# novel tools if it has both the inventory and the capacity?
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/D96_STATUS.txt
sed -e "s|path: data/merged-v2/train.jsonl|path: data/distill/wide.jsonl|" \
    -e "s|out_dir: runs/midtrain-catalog-96m|out_dir: runs/midtrain-distill-96m|" \
    configs/train/midtrain-catalog-96m.yaml > configs/train/midtrain-distill-96m.yaml
sed -e "s|path: data/merged-v2/train.jsonl|path: data/distill/wide.jsonl|" \
    -e "s|init_from: runs/midtrain-catalog-96m/latest.pt|init_from: runs/midtrain-distill-96m/latest.pt|" \
    -e "s|out_dir: runs/sft-catalog-96m|out_dir: runs/sft-distill-96m|" \
    configs/train/sft-catalog-96m.yaml > configs/train/sft-distill-96m.yaml
.venv/bin/localagent train midtrain configs/train/midtrain-distill-96m.yaml \
  > explog/d96_midtrain.log 2>&1
echo "midtrain rc=$?" >> "$STATUS"
.venv/bin/localagent train sft configs/train/sft-distill-96m.yaml > explog/d96_sft.log 2>&1
echo "sft rc=$?" >> "$STATUS"
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-distill-96m/latest.pt \
  --rows 200 --device cuda --out runs/evalsuite/distill-96m.json > explog/d96_eval.log 2>&1
echo "eval rc=$?" >> "$STATUS"
.venv/bin/python scripts/androidcontrol_clean.py --model catalog:runs/sft-distill-96m/latest.pt \
  --device cuda --out runs/evalsuite-clean/distill-96m.json > explog/d96_clean.log 2>&1
echo "clean rc=$?" >> "$STATUS"
OMP_NUM_THREADS=4 .venv/bin/python scripts/throughput.py \
  --model catalog:runs/sft-distill-96m/latest.pt --out runs/throughput/distill-96m.json \
  --threads 4 > explog/d96_tp.log 2>&1
echo "throughput rc=$?" >> "$STATUS"
echo D96_DONE >> "$STATUS"

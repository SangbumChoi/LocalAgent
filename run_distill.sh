#!/usr/bin/env bash
# If weights cannot be inherited across shapes, inherit behaviour: relabel the union corpus with
# the strongest fine-tuned open teacher, then run the identical student chain on it.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/DISTILL_STATUS.txt
for _ in $(seq 1 120); do
  grep -q FINETUNE2_ALL_DONE explog/FT2_STATUS.txt 2>/dev/null && break
  sleep 60
done

.venv/bin/python scripts/distill_teacher.py --teacher data/baselines/LFM2-350M \
  --adapter runs/lora/lfm2-350m --out data/distill/train.jsonl --batch-size 48 \
  > explog/distill_corpus.log 2>&1
echo "corpus rc=$?" >> "$STATUS"

sed -e "s|path: data/merged-v2/train.jsonl|path: data/distill/train.jsonl|" \
    -e "s|out_dir: runs/midtrain-catalog-10m|out_dir: runs/midtrain-distill-10m|" \
    configs/train/midtrain-catalog-10m.yaml > configs/train/midtrain-distill-10m.yaml
sed -e "s|path: data/merged-v2/train.jsonl|path: data/distill/train.jsonl|" \
    -e "s|init_from: runs/midtrain-catalog-10m/latest.pt|init_from: runs/midtrain-distill-10m/latest.pt|" \
    -e "s|out_dir: runs/sft-catalog-10m|out_dir: runs/sft-distill-10m|" \
    configs/train/sft-catalog-10m.yaml > configs/train/sft-distill-10m.yaml

.venv/bin/localagent train midtrain configs/train/midtrain-distill-10m.yaml \
  > explog/distill_midtrain.log 2>&1
echo "midtrain rc=$?" >> "$STATUS"
.venv/bin/localagent train sft configs/train/sft-distill-10m.yaml \
  > explog/distill_sft.log 2>&1
echo "sft rc=$?" >> "$STATUS"
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-distill-10m/latest.pt \
  --rows 200 --device cuda --out runs/evalsuite/distill-10m.json \
  > explog/distill_eval.log 2>&1
echo "eval rc=$?" >> "$STATUS"
.venv/bin/python scripts/androidcontrol_clean.py --model catalog:runs/sft-distill-10m/latest.pt \
  --device cuda --out runs/evalsuite-clean/distill-10m.json > explog/distill_clean.log 2>&1
echo "clean rc=$?" >> "$STATUS"
echo DISTILL_DONE >> "$STATUS"

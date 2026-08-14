#!/usr/bin/env bash
# Arms 1 and 3 on face-h100: relabel wide2 with the 4B teacher, then SFT at both pretraining
# budgets (old chain for arm 1, the 2.1B-token chain for arm 3), eval each.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=/home/jovyan/sbchoi/face-pkgs:src PYTHONUNBUFFERED=1
STATUS=explog/PUSH90_STATUS.txt

.venv/bin/python scripts/distill_teacher.py --teacher data/baselines/Qwen3-4B \
  --adapter runs/lora/qwen3-4b --source data/wide2/train.jsonl --rows 60000 \
  --batch-size 24 --out data/distill2/train.jsonl > explog/p90_relabel_face.log 2>&1
echo "relabel-face rc=$?" >> "$STATUS"

sed -e "s|data/distill/wide.jsonl|data/distill2/train.jsonl|" \
    -e "s|out_dir: runs/sft-distill-96m|out_dir: runs/sft-arm1-96m|" \
    configs/train/sft-distill-96m.yaml > configs/train/sft-arm1-96m.yaml
.venv/bin/localagent train sft configs/train/sft-arm1-96m.yaml > explog/p90_arm1_sft.log 2>&1 \
  || { echo "arm1 sft rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-arm1-96m/latest.pt --rows 200 \
  --device cuda --out runs/evalsuite/arm1-96m.json > explog/p90_arm1_eval.log 2>&1
echo "arm1 rc=$?" >> "$STATUS"

sed -e "s|init_from: runs/midtrain-distill-96m/latest.pt|init_from: runs/midtrain-big-96m/latest.pt|" \
    -e "s|data/distill/wide.jsonl|data/distill2/train.jsonl|" \
    -e "s|out_dir: runs/sft-distill-96m|out_dir: runs/sft-arm3-96m|" \
    configs/train/sft-distill-96m.yaml > configs/train/sft-arm3-96m.yaml
.venv/bin/localagent train sft configs/train/sft-arm3-96m.yaml > explog/p90_arm3_sft.log 2>&1 \
  || { echo "arm3 sft rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-arm3-96m/latest.pt --rows 200 \
  --device cuda --out runs/evalsuite/arm3-96m.json > explog/p90_arm3_eval.log 2>&1
echo "arm3 rc=$?" >> "$STATUS"
echo PUSH90_ALL_DONE >> "$STATUS"

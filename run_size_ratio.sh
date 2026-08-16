#!/usr/bin/env bash
# Does the donor/student size ratio explain why projection failed?
#
# The earlier donors were 13-57x the student. Here the student is held at 95.3M and the donor is
# swept from 1.4x to 6.3x, plus a same-family 1.5x donor into the 10.5M student — the only arm
# where architecture, tokenizer and shape family are all held constant and only size differs.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/RATIO_STATUS.txt

arm() {  # <name> <donor> <student-ckpt> <size-tag>
  local name="$1" donor="$2" student="$3" size="$4" start=$SECONDS
  .venv/bin/python scripts/cross_donor_init.py --donor "$donor" --student "$student" \
    --region blocks --out "runs/donor-init/$name/latest.pt" > "explog/r_init_$name.log" 2>&1 \
    || { echo "$name init rc=$?" >> "$STATUS"; return 1; }
  sed -e "s|init_from: .*|init_from: runs/donor-init/$name/latest.pt|" \
      -e "s|out_dir: runs/midtrain-catalog-$size|out_dir: runs/midtrain-$name|" \
      "configs/train/midtrain-catalog-$size.yaml" > "configs/train/midtrain-$name.yaml"
  sed -e "s|init_from: runs/midtrain-catalog-$size/latest.pt|init_from: runs/midtrain-$name/latest.pt|" \
      -e "s|out_dir: runs/sft-catalog-$size|out_dir: runs/sft-$name|" \
      "configs/train/sft-catalog-$size.yaml" > "configs/train/sft-$name.yaml"
  .venv/bin/localagent train midtrain "configs/train/midtrain-$name.yaml" \
    > "explog/r_mid_$name.log" 2>&1 || { echo "$name midtrain rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/localagent train sft "configs/train/sft-$name.yaml" \
    > "explog/r_sft_$name.log" 2>&1 || { echo "$name sft rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-$name/latest.pt" --rows 200 \
    --device cuda --out "runs/evalsuite/$name.json" > "explog/r_eval_$name.log" 2>&1
  echo "$name done rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

S96=runs/ladder-96m-hybrid-seed2026/latest.pt
S10=runs/mix-10m-hybrid-seed2026/latest.pt
# Donor/student ratio sweep at a fixed 95.3M student.
arm ratio-96m-smollm2-135m  data/baselines/SmolLM2-135M-Instruct "$S96" 96m   # 1.4x
arm ratio-96m-lfm25-230m    data/baselines/LFM2.5-230M           "$S96" 96m   # 2.4x
arm ratio-96m-lfm2-350m     data/baselines/LFM2-350M             "$S96" 96m   # 3.7x
arm ratio-96m-qwen3-06b     data/baselines/Qwen3-0.6B            "$S96" 96m   # 6.3x
# Same family, same tokenizer, 1.5x: only the size differs.
arm ratio-10m-own16m        runs/ladder-16m-hybrid-seed2026/latest.pt "$S10" 10m
echo RATIO_DONE >> "$STATUS"

#!/usr/bin/env bash
# Which depth band of a donor carries transferable information?
#
# Transferring every layer at once cannot attribute the outcome to a depth, so each arm projects
# exactly one third of the student's layers and leaves the remaining two thirds at their random
# initialisation. Two donors of different size run the same three bands, so a band effect can be
# told apart from a donor effect.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/BAND_STATUS.txt

arm() {  # <name> <donor> <band>
  local name="$1" donor="$2" band="$3" start=$SECONDS
  .venv/bin/python scripts/cross_donor_init.py --donor "$donor" \
    --student runs/ladder-96m-hybrid-seed2026/latest.pt --region blocks --band "$band" \
    --out "runs/donor-init/$name/latest.pt" > "explog/b_init_$name.log" 2>&1 \
    || { echo "$name init rc=$?" >> "$STATUS"; return 1; }
  sed -e "s|init_from: .*|init_from: runs/donor-init/$name/latest.pt|" \
      -e "s|out_dir: runs/midtrain-catalog-96m|out_dir: runs/midtrain-$name|" \
      configs/train/midtrain-catalog-96m.yaml > "configs/train/midtrain-$name.yaml"
  sed -e "s|init_from: runs/midtrain-catalog-96m/latest.pt|init_from: runs/midtrain-$name/latest.pt|" \
      -e "s|out_dir: runs/sft-catalog-96m|out_dir: runs/sft-$name|" \
      configs/train/sft-catalog-96m.yaml > "configs/train/sft-$name.yaml"
  .venv/bin/localagent train midtrain "configs/train/midtrain-$name.yaml" \
    > "explog/b_mid_$name.log" 2>&1 || { echo "$name midtrain rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/localagent train sft "configs/train/sft-$name.yaml" \
    > "explog/b_sft_$name.log" 2>&1 || { echo "$name sft rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-$name/latest.pt" --rows 200 \
    --device cuda --out "runs/evalsuite/$name.json" > "explog/b_eval_$name.log" 2>&1
  echo "$name done rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

for band in early middle late; do
  arm "band-qwen3-06b-$band" data/baselines/Qwen3-0.6B "$band"
done
for band in early middle late; do
  arm "band-lfm2-350m-$band" data/baselines/LFM2-350M "$band"
done
echo BAND_DONE >> "$STATUS"

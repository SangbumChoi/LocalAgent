#!/usr/bin/env bash
# Replicate the 10.5M catalog chain under fresh seeds. Two nominally identical runs already differ
# by more than several ladder rungs do, so the paper needs a measured noise floor before it reads
# anything into a rung-to-rung gap.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/SEED_SPREAD_STATUS.txt
for seed in "$@"; do
  sed -e "s|out_dir: runs/midtrain-catalog-10m|out_dir: runs/midtrain-seed$seed|" \
      -e "s|^  seed: 2026|  seed: $seed|" \
      configs/train/midtrain-catalog-10m.yaml > "configs/train/midtrain-seed$seed.yaml"
  sed -e "s|init_from: runs/midtrain-catalog-10m/latest.pt|init_from: runs/midtrain-seed$seed/latest.pt|" \
      -e "s|out_dir: runs/sft-catalog-10m|out_dir: runs/sft-seed$seed|" \
      -e "s|^  seed: 2026|  seed: $seed|" \
      configs/train/sft-catalog-10m.yaml > "configs/train/sft-seed$seed.yaml"
  .venv/bin/localagent train midtrain "configs/train/midtrain-seed$seed.yaml" \
    > "explog/seed_midtrain_$seed.log" 2>&1 || { echo "seed$seed midtrain rc=$?" >> "$STATUS"; continue; }
  .venv/bin/localagent train sft "configs/train/sft-seed$seed.yaml" \
    > "explog/seed_sft_$seed.log" 2>&1 || { echo "seed$seed sft rc=$?" >> "$STATUS"; continue; }
  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-seed$seed/latest.pt" \
    --rows 200 --device cuda --out "runs/evalsuite/seed-$seed.json" \
    > "explog/seed_eval_$seed.log" 2>&1
  echo "seed$seed done rc=$?" >> "$STATUS"
done
echo SEED_SPREAD_DONE >> "$STATUS"

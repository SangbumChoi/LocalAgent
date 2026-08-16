#!/usr/bin/env bash
# Region transfer measured on behaviour, not loss: each arm keeps one region of the pretrained
# backbone, then runs the identical catalog midtrain -> SFT -> eval chain.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
DONOR=runs/mix-10m-hybrid-seed2026/latest.pt
STATUS=explog/REGION_CATALOG_STATUS.txt

arm() {
  local region="$1" start=$SECONDS
  .venv/bin/python scripts/region_catalog_init.py --region "$region" --donor "$DONOR" \
    --out "runs/region-init/$region/latest.pt" > "explog/rc_init_$region.log" 2>&1 || {
      echo "$region init rc=$?" >> "$STATUS"; return 1; }

  sed -e "s|init_from: .*|init_from: runs/region-init/$region/latest.pt|" \
      -e "s|out_dir: runs/midtrain-catalog-10m|out_dir: runs/midtrain-rc-$region|" \
      configs/train/midtrain-catalog-10m.yaml > "configs/train/midtrain-rc-$region.yaml"
  sed -e "s|init_from: .*|init_from: runs/midtrain-rc-$region/latest.pt|" \
      -e "s|out_dir: runs/sft-catalog-10m|out_dir: runs/sft-rc-$region|" \
      configs/train/sft-catalog-10m.yaml > "configs/train/sft-rc-$region.yaml"

  .venv/bin/localagent train midtrain "configs/train/midtrain-rc-$region.yaml" \
    > "explog/rc_midtrain_$region.log" 2>&1 || { echo "$region midtrain rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/localagent train sft "configs/train/sft-rc-$region.yaml" \
    > "explog/rc_sft_$region.log" 2>&1 || { echo "$region sft rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-rc-$region/latest.pt" \
    --rows 200 --device cuda --out "runs/evalsuite/rc-$region.json" \
    > "explog/rc_eval_$region.log" 2>&1
  echo "$region done rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

for region in "$@"; do arm "$region"; done
echo REGION_CATALOG_DONE >> "$STATUS"

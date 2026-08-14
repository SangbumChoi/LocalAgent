#!/usr/bin/env bash
# Which open-source checkpoint is the best donor for a differently-shaped student? Same student,
# same chain, same corpus; only the donor and the transferred region vary.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STUDENT=runs/mix-10m-hybrid-seed2026/latest.pt
STATUS=explog/DONOR_STATUS.txt

arm() {
  local donor="$1" tag="$2" region="$3" name="dc-$2-$3" start=$SECONDS
  .venv/bin/python scripts/cross_donor_init.py --donor "data/baselines/$donor" \
    --student "$STUDENT" --region "$region" --out "runs/donor-init/$name/latest.pt" \
    > "explog/dc_init_$name.log" 2>&1 || { echo "$name init rc=$?" >> "$STATUS"; return 1; }
  sed -e "s|init_from: .*|init_from: runs/donor-init/$name/latest.pt|" \
      -e "s|out_dir: runs/midtrain-catalog-10m|out_dir: runs/midtrain-$name|" \
      configs/train/midtrain-catalog-10m.yaml > "configs/train/midtrain-$name.yaml"
  sed -e "s|init_from: .*|init_from: runs/midtrain-$name/latest.pt|" \
      -e "s|out_dir: runs/sft-catalog-10m|out_dir: runs/sft-$name|" \
      configs/train/sft-catalog-10m.yaml > "configs/train/sft-$name.yaml"
  .venv/bin/localagent train midtrain "configs/train/midtrain-$name.yaml" \
    > "explog/dc_midtrain_$name.log" 2>&1 || { echo "$name midtrain rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/localagent train sft "configs/train/sft-$name.yaml" \
    > "explog/dc_sft_$name.log" 2>&1 || { echo "$name sft rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-$name/latest.pt" \
    --rows 200 --device cuda --out "runs/evalsuite/$name.json" \
    > "explog/dc_eval_$name.log" 2>&1
  echo "$name done rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

for pair in SmolLM2-135M-Instruct:smollm2-135m SmolLM2-360M-Instruct:smollm2-360m \
            LFM2-350M:lfm2-350m h2o-danube3-500m-chat:danube3-500m \
            Qwen2.5-0.5B-Instruct:qwen25-05b Qwen3-0.6B:qwen3-06b; do
  arm "${pair%%:*}" "${pair#*:}" blocks
done
for pair in SmolLM2-360M-Instruct:smollm2-360m Qwen2.5-0.5B-Instruct:qwen25-05b \
            Qwen3-0.6B:qwen3-06b; do
  arm "${pair%%:*}" "${pair#*:}" all
done
echo DONOR_SWEEP_DONE >> "$STATUS"

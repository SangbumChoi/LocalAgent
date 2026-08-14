#!/usr/bin/env bash
# Round-3 evidence repair: three seeds per band arm on two models, plus a random-third control,
# so the band table meets the paper's own replicate-spread standard.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/BANDSEED_STATUS.txt

arm() {  # <tag> <base> <band> <seed>
  local tag="$1" base="$2" band="$3" seed="$4" start=$SECONDS
  local name="${tag}-${band}band-s${seed}" layers
  layers=$(.venv/bin/python scripts/pick_band.py runs/analysis/layer_profiles.json "$tag" "$band") \
    || { echo "$name pick rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/finetune_public.py --base "data/baselines/$base" \
    --out "runs/lora/$name" --layers "$layers" --seed "$seed" \
    > "explog/bs_ft_$name.log" 2>&1 || { echo "$name ft rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/eval_suite.py \
    --model "lora:data/baselines/$base|runs/lora/$name" --rows 200 --device cuda \
    --out "runs/evalsuite/band-ft-$name.json" > "explog/bs_ev_$name.log" 2>&1
  echo "$name done rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

for pair in "smollm2-135m:SmolLM2-135M-Instruct" "qwen3-06b:Qwen3-0.6B"; do
  tag="${pair%%:*}"; base="${pair##*:}"
  for band in top bottom; do
    for seed in 101 202; do
      arm "$tag" "$base" "$band" "$seed"
    done
  done
  arm "$tag" "$base" random 2026
done
echo BANDSEED_DONE >> "$STATUS"

#!/usr/bin/env bash
# The causal test of the layer hypothesis: adapt ONLY the top-dW third of layers (as measured on
# that same model), versus only the bottom third, with the all-layer ft-* arm as the ceiling. If
# agentic ability concentrates where the profile says, the top band recovers most of the ceiling
# and the bottom band recovers little, at one third the trainable parameters each.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/CAUSAL_STATUS.txt

.venv/bin/python scripts/analyze_layers.py --out runs/analysis/layer_profiles.json \
  > explog/layer_profiles.log 2>&1
echo "profiles rc=$?" >> "$STATUS"

arm() {  # <tag> <base-dir> <band>
  local tag="$1" base="$2" band="$3" start=$SECONDS
  local layers
  layers=$(.venv/bin/python scripts/pick_band.py runs/analysis/layer_profiles.json "$tag" "$band") \
    || { echo "$tag $band pick rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/finetune_public.py --base "data/baselines/$base" \
    --out "runs/lora/${tag}-${band}band" --layers "$layers" \
    > "explog/c_ft_${tag}_${band}.log" 2>&1 \
    || { echo "$tag $band ft rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/eval_suite.py \
    --model "lora:data/baselines/$base|runs/lora/${tag}-${band}band" --rows 200 --device cuda \
    --out "runs/evalsuite/band-ft-${tag}-${band}.json" > "explog/c_ev_${tag}_${band}.log" 2>&1
  echo "$tag $band done rc=$? layers=$layers secs=$((SECONDS-start))" >> "$STATUS"
}

for pair in "smollm2-135m:SmolLM2-135M-Instruct" "lfm2-350m:LFM2-350M" "qwen3-06b:Qwen3-0.6B"; do
  tag="${pair%%:*}"; base="${pair##*:}"
  arm "$tag" "$base" top
  arm "$tag" "$base" bottom
done
echo CAUSAL_DONE >> "$STATUS"

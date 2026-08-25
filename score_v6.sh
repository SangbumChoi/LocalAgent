#!/usr/bin/env bash
# Six suites now. Worker "new" fine-tunes and scores the three added models; worker "rescore"
# brings every arm already in the paper onto the same suite set.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/V6_STATUS.txt
score() { local tag="$1" spec="$2" s=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "$spec" --rows 200 --device cuda \
    --out "runs/evalsuite/$tag.json" > "explog/v6_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-s))" >> "$STATUS"; }

if [ "$1" = "new" ]; then
  .venv/bin/python scripts/chance_baseline.py > explog/chance6.log 2>&1
  echo "chance rc=$?" >> "$STATUS"
  for pair in LFM2.5-230M:lfm25-230m granite-4.0-h-350m:granite-h-350m \
              granite-4.0-350m:granite-350m; do
    base="data/baselines/${pair%%:*}"; tag="${pair#*:}"
    score "$tag" "hf:$base"
    .venv/bin/python scripts/finetune_public.py --base "$base" --out "runs/lora/$tag" \
      --steps 600 > "explog/ft6_$tag.log" 2>&1
    echo "ft-$tag rc=$?" >> "$STATUS"
    score "ft-$tag" "lora:$base|runs/lora/$tag"
    OMP_NUM_THREADS=4 .venv/bin/python scripts/throughput.py --model "hf:$base" \
      --out "runs/throughput/$tag.json" --threads 4 > "explog/tp6_$tag.log" 2>&1
    echo "tp-$tag rc=$?" >> "$STATUS"
  done
  echo NEW_MODELS_DONE >> "$STATUS"
else
  for size in 10m 96m; do score "catalog-$size" "catalog:runs/sft-catalog-$size/latest.pt"; done
  score wide-10m catalog:runs/sft-wide-10m/latest.pt
  score distill-10m catalog:runs/sft-distill2-10m/latest.pt
  score distill-96m catalog:runs/sft-distill-96m/latest.pt
  score profiled-10m catalog:runs/sft-profiled-10m/latest.pt
  for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2-350M:lfm2-350m \
              SmolLM2-360M-Instruct:smollm2-360m h2o-danube3-500m-chat:danube3-500m \
              Qwen2.5-0.5B-Instruct:qwen25-05b Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b \
              LFM2-700M:lfm2-700m Qwen3-0.6B:qwen3-06b; do
    base="data/baselines/${pair%%:*}"; tag="${pair#*:}"
    score "$tag" "hf:$base"
    score "ft-$tag" "lora:$base|runs/lora/$tag"
  done
  echo RESCORE_V6_DONE >> "$STATUS"
fi

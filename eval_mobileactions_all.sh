#!/usr/bin/env bash
# Re-score mobileactions for every model after the schema-casing normalisation; results merge
# into each model's main receipt.
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src
ev() { .venv/bin/python scripts/eval_suite.py --model "$1" --rows 200 --device cuda --suites mobileactions --out "$2" >> explog/ma_resweep.log 2>&1 && echo "ma $2 done" >> explog/MA_STATUS.txt; }
for arm in sft-arm2-96m:arm2-96m sft-arm3-96m:arm3-96m sft-fresh-96m:fresh-96m sft-big-96m:big-96m sft-skel-96m:skel-96m sft-anti-96m:anti-96m sft-embed-96m:embed-96m sft-fullproj-96m:fullproj-96m sft-rand-96m:rand-96m sft-skel2-96m:skel2-96m sft-rand2-96m:rand2-96m sft-skel-fresh-96m:skel-fresh-96m sft-wide3-96m:wide3-96m sft-catalog-10m:catalog-10m sft-catalog-16m:catalog-16m sft-catalog-96m:catalog-96m sft-distill2-10m:distill-10m sft-distill-96m:distill-96m; do
  ckpt="runs/${arm%%:*}/latest.pt"; out="runs/evalsuite/${arm##*:}.json"
  [ -f "$ckpt" ] && ev "catalog:$ckpt" "$out"
done
DIR2TAG() { case "$1" in SmolLM2-135M-Instruct) echo smollm2-135m;; LFM2.5-230M) echo lfm25-230m;; LFM2-350M) echo lfm2-350m;; granite-4.0-h-350m) echo granite-h-350m;; granite-4.0-350m) echo granite-350m;; SmolLM2-360M-Instruct) echo smollm2-360m;; h2o-danube3-500m-chat) echo danube3-500m;; Qwen2.5-0.5B-Instruct) echo qwen25-05b;; Qwen2.5-Coder-0.5B-Instruct) echo qwen25-coder-05b;; LFM2-700M) echo lfm2-700m;; Qwen3-0.6B) echo qwen3-06b;; *) echo "";; esac; }
for d in data/baselines/*/; do
  name=$(basename "$d"); [ "$name" = gemma-3-270m-it ] && continue
  tag=$(DIR2TAG "$name")
  [ -f "$d/config.json" ] || continue
  if [ -n "$tag" ]; then relout="runs/evalsuite/$tag.json"; ftout="runs/evalsuite/ft-$tag.json"; else relout="runs/evalsuite/rel-$name.json"; ftout="runs/evalsuite/ft-$name.json"; fi
  [ -f "$relout" ] && ev "hf:$d" "$relout"
  [ -d "runs/lora/$name" ] && [ -f "$ftout" ] && ev "lora:$d|runs/lora/$name" "$ftout"
done
echo MA-RESWEEP-DONE >> explog/MA_STATUS.txt

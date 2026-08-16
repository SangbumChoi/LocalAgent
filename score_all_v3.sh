#!/usr/bin/env bash
# One harness version, five suites, every arm. Split into two workers so the public models (slow)
# and our own checkpoints (fast) do not serialise behind each other.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/V3_STATUS.txt
score() {
  local tag="$1" spec="$2" start=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "$spec" --rows 200 --device cuda \
    --out "runs/evalsuite/$tag.json" > "explog/v3_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

if [ "$1" = "ours" ]; then
  .venv/bin/python scripts/majority_baseline.py > explog/majority.log 2>&1
  echo "majority rc=$?" >> "$STATUS"
  for size in 10m 16m 35m 96m; do score "catalog-$size" "catalog:runs/sft-catalog-$size/latest.pt"; done
  for region in scratch norms embed attn ffn early late no_embed full; do
    score "rc-$region" "catalog:runs/sft-rc-$region/latest.pt"
  done
  for seed in 101 202 303; do score "seed-$seed" "catalog:runs/sft-seed$seed/latest.pt"; done
  score dispatch dispatch:
  echo OURS_V3_DONE >> "$STATUS"
else
  for pair in SmolLM2-135M-Instruct:smollm2-135m SmolLM2-360M-Instruct:smollm2-360m \
              Qwen2.5-0.5B-Instruct:qwen25-05b Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b \
              Qwen3-0.6B:qwen3-06b LFM2-350M:lfm2-350m LFM2-700M:lfm2-700m \
              h2o-danube3-500m-chat:danube3-500m; do
    score "${pair#*:}" "hf:data/baselines/${pair%%:*}"
  done
  echo PUBLIC_V3_DONE >> "$STATUS"
fi

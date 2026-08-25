#!/usr/bin/env bash
# Re-score every arm with one parser version: the parser now accepts the Pythonic call format,
# so mixing old and new numbers in the same table would compare scoring rules, not models.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/EVALSUITE_STATUS.txt
score() {
  local tag="$1" spec="$2" start=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "$spec" --rows 200 --device cuda \
    --out "runs/evalsuite/$tag.json" > "explog/evalsuite_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start)) [parser-v2]" >> "$STATUS"
}
for size in 10m 16m 35m 96m; do score "catalog-$size" "catalog:runs/sft-catalog-$size/latest.pt"; done
score dispatch dispatch:
for name in SmolLM2-135M-Instruct:smollm2-135m SmolLM2-360M-Instruct:smollm2-360m \
            Qwen2.5-0.5B-Instruct:qwen25-05b LFM2-350M:lfm2-350m Qwen3-0.6B:qwen3-06b; do
  score "${name#*:}" "hf:data/baselines/${name%%:*}"
done
echo RESCORE_V2_DONE >> "$STATUS"

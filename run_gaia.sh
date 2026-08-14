#!/usr/bin/env bash
# GAIA closed-book lower bound: released vs fine-tuned pairs, plus the scale series.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/GAIA_STATUS.txt

run() {  # <tag> <spec>
  local tag="$1" spec="$2" start=$SECONDS
  .venv/bin/python scripts/gaia_eval.py --model "$spec" --data data/public/gaia-validation.jsonl \
    --out "runs/gaia/$tag.json" > "explog/gaia_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

for pair in "qwen3-06b:Qwen3-0.6B" "qwen25-05b:Qwen2.5-0.5B-Instruct" "lfm2-700m:LFM2-700M" \
            "granite-h-350m:granite-4.0-h-350m" "smollm2-360m:SmolLM2-360M-Instruct"; do
  tag="${pair%%:*}"; dir="${pair##*:}"
  run "$tag"    "hf:data/baselines/$dir"
  run "ft-$tag" "lora:data/baselines/$dir|runs/lora/$tag"
done
run ft-qwen3-17b "lora:data/baselines/Qwen3-1.7B|runs/lora/qwen3-17b"
run ft-qwen3-4b  "lora:data/baselines/Qwen3-4B|runs/lora/qwen3-4b"
echo GAIA_DONE >> "$STATUS"

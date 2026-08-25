#!/usr/bin/env bash
# Score each ladder rung as its SFT finishes, plus the newly downloaded sub-1B baselines.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/EVALSUITE_STATUS.txt

score() {  # <tag> <spec>
  local tag="$1" spec="$2"
  local start=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "$spec" --rows 200 --device cuda \
    --out "runs/evalsuite/$tag.json" > "explog/evalsuite_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

score catalog-10m catalog:runs/sft-catalog-10m/latest.pt

for size in 16m 35m 96m; do
  for _ in $(seq 1 90); do
    grep -q "sft-catalog-${size} rc=0" explog/LADDER_STATUS.txt 2>/dev/null && break
    sleep 60
  done
  [ -f "runs/sft-catalog-${size}/latest.pt" ] && score "catalog-${size}" "catalog:runs/sft-catalog-${size}/latest.pt"
done

for _ in $(seq 1 40); do grep -q BASELINES2_DONE explog/baselines2.log 2>/dev/null && break; sleep 30; done
[ -d data/baselines/LFM2-350M ] && score lfm2-350m hf:data/baselines/LFM2-350M
[ -d data/baselines/Qwen3-0.6B ] && score qwen3-06b hf:data/baselines/Qwen3-0.6B
echo LADDER_EVAL_DONE >> "$STATUS"

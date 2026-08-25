#!/usr/bin/env bash
# One standardized evaluation process across model families: sub-1B instruct baselines, the
# repository's byte-level arms, and the deployed retrieve-and-ground dispatcher.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
mkdir -p runs/evalsuite explog
STATUS=explog/EVALSUITE_STATUS.txt
: > "$STATUS"
ROWS=200

run() {  # <tag> <model-spec> [device]
  local tag="$1" spec="$2" device="${3:-cuda}"
  local start=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "$spec" --rows "$ROWS" --device "$device" \
    --out "runs/evalsuite/$tag.json" > "explog/evalsuite_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

( run dispatch dispatch:retrieve+ground cpu ) &
( run smollm2-135m hf:data/baselines/SmolLM2-135M-Instruct ) &
( run smollm2-360m hf:data/baselines/SmolLM2-360M-Instruct ) &
( run qwen25-05b hf:data/baselines/Qwen2.5-0.5B-Instruct ) &
wait
echo BASELINES_SCORED >> "$STATUS"

( run localagent-union localagent:runs/region/data-union/model.pt ) &
( run localagent-scratch localagent:runs/region/region-scratch/model.pt ) &
( run localagent-embed localagent:runs/region/region-embed/model.pt ) &
( run localagent-synthetic localagent:runs/region/data-synthetic/model.pt ) &
wait
echo EVALSUITE_DONE >> "$STATUS"

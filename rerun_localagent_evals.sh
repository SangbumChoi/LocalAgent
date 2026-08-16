#!/usr/bin/env bash
# Re-score every byte-agent arm with the corrected text budget (their tokens are bytes).
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/EVALSUITE_STATUS.txt
pkill -f "eval_suite.py --model localagent" 2>/dev/null
sleep 2

run() {
  local tag="$1" ckpt="$2"
  [ -f "$ckpt" ] || return 0
  local start=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "localagent:$ckpt" --rows 200 --device cuda \
    --out "runs/evalsuite/$tag.json" > "explog/evalsuite_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start)) [byte-budget-fix]" >> "$STATUS"
}

( run localagent-union     runs/region/data-union/model.pt ) &
( run localagent-synthetic runs/region/data-synthetic/model.pt ) &
( run localagent-scratch   runs/region/region-scratch/model.pt ) &
( run localagent-embed     runs/region/region-embed/model.pt ) &
wait
( run localagent-attn      runs/region/region-attn/model.pt ) &
( run localagent-ffn       runs/region/region-ffn/model.pt ) &
( run localagent-early     runs/region/region-early/model.pt ) &
( run localagent-toolace   runs/region/data-toolace/model.pt ) &
wait
echo LOCALAGENT_RESCORED >> "$STATUS"

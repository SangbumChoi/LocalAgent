#!/usr/bin/env bash
# Second evaluation wave on the GPU: every remaining region and mixing arm, same harness and rows.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/EVALSUITE_STATUS.txt
while ! grep -q EVALSUITE_DONE "$STATUS" 2>/dev/null; do sleep 60; done

run() {
  local tag="$1" ckpt="$2"
  [ -f "$ckpt" ] || return 0
  local start=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "localagent:$ckpt" --rows 200 --device cuda \
    --out "runs/evalsuite/$tag.json" > "explog/evalsuite_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

( run localagent-attn runs/region/region-attn/model.pt ) &
( run localagent-ffn runs/region/region-ffn/model.pt ) &
( run localagent-early runs/region/region-early/model.pt ) &
( run localagent-late runs/region/region-late/model.pt ) &
wait
( run localagent-no_embed runs/region/region-no_embed/model.pt ) &
( run localagent-toolace runs/region/data-toolace/model.pt ) &
( run localagent-android runs/region/data-androidcontrol/model.pt ) &
( run localagent-mix-balanced runs/region/mix-balanced/model.pt ) &
wait
echo EVALSUITE_WAVE2_DONE >> "$STATUS"

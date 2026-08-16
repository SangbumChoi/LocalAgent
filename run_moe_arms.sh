#!/usr/bin/env bash
# MoE region transfer. The donor is frozen at its step-400 checkpoint so the arms read a stable file.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/REGION_STATUS.txt
DONOR=runs/moe-donor-frozen.pt
[ -f "$DONOR" ] || cp runs/moe-donor-seed2026/latest.pt "$DONOR"
pkill -f "pretrain-moe-donor" 2>/dev/null
while ! grep -q REGION_GRID_DONE "$STATUS" 2>/dev/null; do sleep 60; done

arm() {
  local region="$1"
  local start=$SECONDS
  .venv/bin/python scripts/moe_region.py --region "$region" --donor "$DONOR" --steps 300 \
    --out "runs/moe-region/$region" > "explog/moe_$region.log" 2>&1
  echo "moe-$region rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

( arm scratch ) & ( arm router ) & ( arm expert0 ) &
wait
( arm experts ) & ( arm attn ) & ( arm full ) &
wait
echo MOE_ARMS_DONE >> "$STATUS"

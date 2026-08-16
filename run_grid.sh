#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
mkdir -p explog runs/region
STATUS=explog/REGION_STATUS.txt
: > "$STATUS"

arm() {  # <tag> <region> <data>
  local tag="$1" region="$2" data="$3"
  local start=$SECONDS
  .venv/bin/python scripts/region_transfer.py --region "$region" --data "$data" \
    --out "runs/region/$tag" --steps 800 --n-synth 2500 --n-public 4000 \
    > "explog/region_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

# Wave 1 — dataset contribution at a fixed full warm start
( arm data-synthetic      full synthetic ) &
( arm data-toolace        full toolace ) &
( arm data-mind2web       full mind2web ) &
( arm data-androidcontrol full androidcontrol ) &
( arm data-union          full union ) &
wait
echo WAVE1_DONE >> "$STATUS"

# Wave 2 — region transfer at the best-covered data mix
( arm region-scratch    scratch    union ) &
( arm region-embed      embed      union ) &
( arm region-norms      norms      union ) &
( arm region-attn       attn       union ) &
( arm region-ffn        ffn        union ) &
wait
echo WAVE2_DONE >> "$STATUS"

( arm region-early      early      union ) &
( arm region-late       late       union ) &
( arm region-attn_embed attn_embed union ) &
( arm region-no_embed   no_embed   union ) &
wait
echo REGION_GRID_DONE >> "$STATUS"

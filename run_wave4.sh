#!/usr/bin/env bash
# Mixing-ratio arms: the union nearly matches every specialist on single-step surfaces but
# collapses on multi-turn trajectories. These arms test whether the collapse is a mixing artefact.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/REGION_STATUS.txt
while ! grep -q REGION_GRID_DONE "$STATUS" 2>/dev/null; do sleep 60; done

arm() {  # <tag> <n_public> <mt_weight>
  local tag="$1" cap="$2" weight="$3"
  local start=$SECONDS
  .venv/bin/python scripts/region_transfer.py --region full --data union \
    --out "runs/region/$tag" --steps 800 --n-synth 2500 --n-public "$cap" --mt-weight "$weight" \
    > "explog/region_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

( arm mix-balanced      1000 1.0 ) &
( arm mix-mtw3          4000 3.0 ) &
( arm mix-balanced-mtw3 1000 3.0 ) &
wait
echo WAVE4_DONE >> "$STATUS"

#!/usr/bin/env bash
# Standing scheduler: pop queue head after 3 consecutive idle 2-min checks; log every iteration.
cd /home/jovyan/sbchoi/localagent || exit 1
QUEUE="$1"; REG=experiments/registry.log; DBG=experiments/daemon-$(hostname).log
mkdir -p experiments
idle=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  busy=0
  pgrep -f "localagent[ ]train|distill[_]teacher|eval[_]suite|finetune[_]public|skeleton[_]arm" >/dev/null && busy=1
  if [ "$busy" = 1 ]; then idle=0
  elif [ "${util:-100}" -lt 5 ] && [ "${mem:-99999}" -lt 2000 ]; then idle=$((idle+1)); else idle=0; fi
  echo "$(date -u +%FT%TZ) util=$util mem=$mem busy=$busy idle=$idle qlines=$(wc -l < "$QUEUE" 2>/dev/null)" >> "$DBG"
  if [ "$idle" -ge 3 ] && [ -s "$QUEUE" ]; then
    job=$(head -1 "$QUEUE"); sed -i '1d' "$QUEUE"
    if [ -n "$job" ]; then
      echo "$(date -u +%FT%TZ) START $(hostname) :: $job" >> "$REG"
      bash -c "$job" >> "$DBG" 2>&1
      echo "$(date -u +%FT%TZ) EXIT=$? $(hostname) :: $job" >> "$REG"
      idle=0
    fi
  fi
  sleep 120
done

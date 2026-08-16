#!/usr/bin/env bash
# The fixed-head arms carry Table 1, so they need the same five suites as everything else.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
for pair in localagent-union:runs/region/data-union/model.pt \
            localagent-scratch:runs/region/region-scratch/model.pt; do
  tag="${pair%%:*}"; path="${pair#*:}"; start=$SECONDS
  .venv/bin/python scripts/eval_suite.py --model "localagent:$path" --rows 200 --device cuda \
    --out "runs/evalsuite/$tag.json" > "explog/v3_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> explog/V3_STATUS.txt
done
echo FIXEDHEAD_V3_DONE >> explog/V3_STATUS.txt

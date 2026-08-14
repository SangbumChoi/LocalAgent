#!/usr/bin/env bash
# Score one unchanged checkpoint three times. Two runs of the same weights already disagreed by
# more than the gaps this report wants to interpret, so the noise floor has to be measured, not
# assumed away.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
for run in 1 2 3; do
  .venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-catalog-10m/latest.pt \
    --rows 200 --device cuda --out "runs/evalsuite/repeat-$run.json" \
    > "explog/repeat_$run.log" 2>&1
  echo "repeat-$run rc=$?" >> explog/REPEAT_STATUS.txt
done
echo REPEAT_DONE >> explog/REPEAT_STATUS.txt

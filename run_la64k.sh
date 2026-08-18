#!/usr/bin/env bash
# One 64k-vocab variant end to end: pretrain (2.1B tokens) -> midtrain -> SFT -> eval -> throughput.
#   bash run_la64k.sh 8k   |   bash run_la64k.sh 16k
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 LOCALAGENT_RESUME_LINEAGE=warn
tag="$1"; STATUS=explog/LA64K_STATUS.txt

# STAGE GUARDS: a stage that has recorded success is never re-entered, so relaunches resume
# at the first unfinished stage instead of re-running (or re-breaking) finished ones.
t0=$SECONDS
if ! grep -q "la64k-${tag} pretrain secs=" "$STATUS"; then
  .venv/bin/localagent train pretrain "configs/train/pretrain-la64k-${tag}.yaml" \
    > "explog/l64_pre_${tag}.log" 2>&1 || { echo "la64k-${tag} pretrain rc=$?" >> "$STATUS"; exit 1; }
  echo "la64k-${tag} pretrain secs=$((SECONDS-t0))" >> "$STATUS"
fi
if ! grep -q "la64k-${tag} midtrain secs=" "$STATUS"; then
  t1=$SECONDS
  .venv/bin/localagent train midtrain "configs/train/midtrain-la64k-${tag}.yaml" \
    > "explog/l64_mid_${tag}.log" 2>&1 || { echo "la64k-${tag} midtrain rc=$?" >> "$STATUS"; exit 1; }
  echo "la64k-${tag} midtrain secs=$((SECONDS-t1))" >> "$STATUS"
fi
if ! grep -q "la64k-${tag} sft secs=" "$STATUS"; then
  t2=$SECONDS
  .venv/bin/localagent train sft "configs/train/sft-la64k-${tag}.yaml" \
    > "explog/l64_sft_${tag}.log" 2>&1 || { echo "la64k-${tag} sft rc=$?" >> "$STATUS"; exit 1; }
  echo "la64k-${tag} sft secs=$((SECONDS-t2))" >> "$STATUS"
fi
.venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-la64k-${tag}/latest.pt" \
  --rows 200 --device cuda --out "runs/evalsuite/la64k-${tag}.json" \
  > "explog/l64_eval_${tag}.log" 2>&1
echo "la64k-${tag} eval rc=$?" >> "$STATUS"
.venv/bin/python scripts/throughput.py --model "catalog:runs/sft-la64k-${tag}/latest.pt" \
  --out "runs/throughput/la64k-${tag}.json" > "explog/l64_tp_${tag}.log" 2>&1
echo "la64k-${tag} throughput rc=$? DONE" >> "$STATUS"

#!/usr/bin/env bash
# Qwen3 defaults to a thinking turn that consumes the whole 64-token budget before any call is
# emitted; re-score it on the non-thinking path so the number reflects the model, not the mode.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
for _ in $(seq 1 60); do
  grep -q REPEAT_DONE explog/REPEAT_STATUS.txt 2>/dev/null && break
  sleep 30
done
.venv/bin/python scripts/eval_suite.py --model hf:data/baselines/Qwen3-0.6B --rows 200 \
  --device cuda --out runs/evalsuite/qwen3-06b.json > explog/evalsuite_qwen3-06b.log 2>&1
echo "qwen3-06b rc=$? [no-thinking]" >> explog/EVALSUITE_STATUS.txt

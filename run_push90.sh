#!/usr/bin/env bash
# The honest ≥90 campaign, stage 1 on thor2: error decomposition, wide2 corpus with ToolACE
# coverage, teacher relabelling by the strongest teacher (ft-Qwen3-4B), then the coverage+teacher
# arm at the unchanged pretraining budget. No evaluation-only suite ever enters training; every
# added split passes the same prompt-hash guard the wide corpus used.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/PUSH90_STATUS.txt

.venv/bin/python scripts/analyze_errors.py --model catalog:runs/sft-distill-96m/latest.pt \
  --out runs/analysis/errors-distill-96m.json > explog/p90_errors_ours.log 2>&1
echo "errors-ours rc=$?" >> "$STATUS"
.venv/bin/python scripts/analyze_errors.py --model "lora:data/baselines/Qwen3-0.6B|runs/lora/qwen3-06b" \
  --out runs/analysis/errors-ft-qwen3-06b.json > explog/p90_errors_ft.log 2>&1
echo "errors-ft rc=$?" >> "$STATUS"

.venv/bin/python scripts/build_wide_corpus.py --base data/wide/train.jsonl \
  --add data/public/toolace-train.jsonl --cap 18000 --out data/wide2/train.jsonl \
  --guard data/public/toolace-eval.jsonl --guard data/public/xlam-test.jsonl \
  --guard data/public/bfcl-eval.jsonl --guard data/public/toolbench-eval.jsonl \
  --guard data/public/agentnet-eval.jsonl --guard data/public/androidcontrol-test.jsonl \
  --guard data/merged-v2/eval-mind2web.jsonl > explog/p90_wide2.log 2>&1
echo "wide2 rc=$?" >> "$STATUS"

.venv/bin/python scripts/distill_teacher.py --teacher data/baselines/Qwen3-4B \
  --adapter runs/lora/qwen3-4b --source data/wide2/train.jsonl --rows 60000 \
  --batch-size 24 --out data/distill2/train.jsonl > explog/p90_relabel.log 2>&1
echo "relabel rc=$?" >> "$STATUS"

# Arm 1: coverage + teacher at the unchanged pretraining budget.
sed -e "s|data/distill/wide.jsonl|data/distill2/train.jsonl|" \
    -e "s|out_dir: runs/sft-distill-96m|out_dir: runs/sft-arm1-96m|" \
    configs/train/sft-distill-96m.yaml > configs/train/sft-arm1-96m.yaml
.venv/bin/localagent train sft configs/train/sft-arm1-96m.yaml > explog/p90_arm1_sft.log 2>&1 \
  || { echo "arm1 sft rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-arm1-96m/latest.pt --rows 200 \
  --device cuda --out runs/evalsuite/arm1-96m.json > explog/p90_arm1_eval.log 2>&1
echo "arm1 rc=$?" >> "$STATUS"
echo PUSH90_STAGE1_DONE >> "$STATUS"

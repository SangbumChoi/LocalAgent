#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/run_gpu_campaign.py \
  --output runs/gpu-campaign-transfer \
  --device cuda --skip-benchmark --skip-matrix --force \
  --transfer runs/ablate/cuda-s0/model.pt:runs/ablate/cuda-s0-x3/model.pt \
  --checkpoint runs/ablate/cuda-s0-x3/model.pt 2>&1 | tail -20
echo "TRANSFER_RC=$?"

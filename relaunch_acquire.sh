#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
pgrep -af "download_pretrain_mixture" | head -2
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=1800 HF_HUB_ETAG_TIMEOUT=1800
.venv/bin/python scripts/run_gpu_campaign.py --output runs/hf-acquire --skip-benchmark --skip-matrix --acquire-hf --hf-out data/hf-campaign --force \
  --hf-source toolace --hf-source android_control_public_mirror --hf-source localagent_tiny_model --hf-source mind2web_train --hf-source agentnet \
  > explog/hf_acquire.log 2>&1
echo HF_ACQ_RC=$? >> explog/hf_acquire.log' </dev/null >/dev/null 2>&1 &
sleep 3; echo acquire_relaunched

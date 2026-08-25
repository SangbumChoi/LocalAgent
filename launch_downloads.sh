#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
export HF_ENDPOINT=http://127.0.0.1:8899
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY}"
export HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_ETAG_TIMEOUT=900

setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ENABLE_HF_TRANSFER=0
.venv/bin/python scripts/run_gpu_campaign.py --output runs/hf-acquire --skip-benchmark --skip-matrix --acquire-hf --hf-out data/hf-campaign --force \
  --hf-source toolace --hf-source mind2web_train --hf-source android_control_public_mirror --hf-source agentnet --hf-source localagent_tiny_model \
  > explog/hf_acquire.log 2>&1
echo HF_ACQ_RC=$? >> explog/hf_acquire.log' </dev/null >/dev/null 2>&1 &

setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ENABLE_HF_TRANSFER=0
.venv/bin/python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml --out data/raw/paper --resume \
  --license-evidence smollm-card=data/provenance/smollm.md \
  --license-evidence codeparrot-card=data/provenance/codeparrot.md \
  --license-evidence websight-card=data/provenance/websight.md \
  --plan-out explog/corpus_download_plan.json > explog/corpus_download.log 2>&1
echo CORPUS_DL_RC=$? >> explog/corpus_download.log' </dev/null >/dev/null 2>&1 &
sleep 3
echo "launched"; pgrep -cf "run_gpu_campaign|download_pretrain_mixture"

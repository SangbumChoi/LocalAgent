#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
pgrep -f "download_pretrain_mixture" >/dev/null && { echo "already running"; exit 0; }
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=1800 HF_HUB_ETAG_TIMEOUT=1800
: > explog/corpus_download.log
for i in $(seq 1 12); do
  .venv/bin/python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml --out data/raw/paper --resume \
    --license-evidence smollm-card=data/provenance/smollm.md \
    --license-evidence codeparrot-card=data/provenance/codeparrot.md \
    --license-evidence websight-card=data/provenance/websight.md \
    --plan-out explog/corpus_download_plan.json >> explog/corpus_download.log 2>&1 && { echo CORPUS_OK >> explog/corpus_download.log; break; }
  echo "RETRY $i" >> explog/corpus_download.log; sleep 20
done
echo CORPUS_FINISHED >> explog/corpus_download.log' </dev/null >/dev/null 2>&1 &
sleep 3; echo corpus_relaunched

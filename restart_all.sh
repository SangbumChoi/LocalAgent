#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
rm -rf hf_api_snapshot && tar xzf ../hf_snapshot2.tgz
pkill -f "hf[_]api_shim" 2>/dev/null; sleep 1
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot --port 8899 </dev/null > explog/shim.log 2>&1 &
sleep 4
echo -n "shim_tree_entries="; curl -s -m 20 --noproxy '*' 'http://127.0.0.1:8899/api/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f' | python3 -c "import json,sys; print(len(json.load(sys.stdin)))"
rm -rf data/raw/paper
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ENABLE_HF_TRANSFER=0
.venv/bin/python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml --out data/raw/paper \
  --license-evidence smollm-card=data/provenance/smollm.md \
  --license-evidence codeparrot-card=data/provenance/codeparrot.md \
  --license-evidence websight-card=data/provenance/websight.md \
  --plan-out explog/corpus_download_plan.json > explog/corpus_download.log 2>&1
echo CORPUS_DL_RC=$? >> explog/corpus_download.log' </dev/null >/dev/null 2>&1 &
sleep 2; echo "corpus relaunched"
echo "--- ablate x3 error ---"; tail -4 explog/ablate_cuda-s0-x3.log
echo "--- hf acquire ---"; tail -2 explog/hf_acquire.log
echo "--- cpu ablation ---"; tail -2 explog/ablate_cpu-s0.log

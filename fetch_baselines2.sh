#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
rm -rf hf_api_snapshot && tar xzf ../hf_snapshot4.tgz
pkill -f "hf[_]api_shim" 2>/dev/null; sleep 1
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot --cache /home/jovyan/sbchoi/hfcache --port 8899 </dev/null > explog/shim.log 2>&1 &
sleep 4
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DISABLE_XET=1
: > explog/baselines2.log
.venv/bin/pip show transformers 2>/dev/null | head -2 >> explog/baselines2.log
.venv/bin/python - <<PY >> explog/baselines2.log 2>&1
from huggingface_hub import snapshot_download
for repo in ["LiquidAI/LFM2-350M", "Qwen/Qwen3-0.6B"]:
    path = snapshot_download(repo, local_dir=f"data/baselines/{repo.split(chr(47))[-1]}",
                             allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"])
    print("downloaded", repo, path, flush=True)
PY
echo BASELINES2_DONE >> explog/baselines2.log' </dev/null >/dev/null 2>&1 &
sleep 3; echo baselines2_launched

#!/usr/bin/env bash
# New comparison models and three more open agent benchmarks.
cd /home/jovyan/sbchoi/localagent || exit 1
rm -rf hf_api_snapshot_v5 && tar xzf ../hf_snapshot5.tgz
pkill -f "hf[_]api_shim" 2>/dev/null; sleep 1
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot_v5 \
  --cache /home/jovyan/sbchoi/hfcache --port 8899 </dev/null > explog/shim5.log 2>&1 &
sleep 5
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DISABLE_XET=1
: > explog/fetch_v5.log
.venv/bin/python - <<PY >> explog/fetch_v5.log 2>&1
from huggingface_hub import snapshot_download
for repo in ["LiquidAI/LFM2.5-230M", "ibm-granite/granite-4.0-h-350m",
             "ibm-granite/granite-4.0-350m"]:
    try:
        snapshot_download(repo, local_dir=f"data/baselines/{repo.split(chr(47))[-1]}",
                          allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "*.model"],
                          ignore_patterns=["*consolidated*", "onnx/*", "*.gguf"])
        print("OK", repo, flush=True)
    except Exception as e:
        print("FAIL", repo, type(e).__name__, str(e)[:160], flush=True)
for repo in ["osunlp/Mind2Web", "liminghao1630/API-Bank",
             "gorilla-llm/Berkeley-Function-Calling-Leaderboard"]:
    try:
        snapshot_download(repo, repo_type="dataset",
                          local_dir=f"data/hf-campaign/{repo.split(chr(47))[-1]}",
                          allow_patterns=["*.json", "*.jsonl", "*.parquet", "*.csv"],
                          ignore_patterns=["*train_*.zip", "*.tar*", "images/*"])
        print("OK", repo, flush=True)
    except Exception as e:
        print("FAIL", repo, type(e).__name__, str(e)[:160], flush=True)
print("FETCH_V5_DONE", flush=True)
PY' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

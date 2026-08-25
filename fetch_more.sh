#!/usr/bin/env bash
# Widen the public-model comparison and add xLAM's official test split as a second
# function-calling surface. Metadata comes from the replayed snapshot; bytes from the mirror.
cd /home/jovyan/sbchoi/localagent || exit 1
rm -rf hf_api_snapshot_v4 && tar xzf ../hf_snapshot4b.tgz
pkill -f "hf[_]api_shim" 2>/dev/null; sleep 1
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot_v4 \
  --cache /home/jovyan/sbchoi/hfcache --port 8899 </dev/null > explog/shim4.log 2>&1 &
sleep 5
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DISABLE_XET=1
: > explog/more_baselines.log
.venv/bin/python - <<PY >> explog/more_baselines.log 2>&1
from huggingface_hub import snapshot_download
MODELS = ["Qwen/Qwen2.5-Coder-0.5B-Instruct", "LiquidAI/LFM2-700M",
          "h2oai/h2o-danube3-500m-chat", "google/gemma-3-270m-it"]
for repo in MODELS:
    try:
        snapshot_download(repo, local_dir=f"data/baselines/{repo.split(chr(47))[-1]}",
                          allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "*.model"],
                          ignore_patterns=["*consolidated*", "onnx/*", "*.gguf"])
        print("OK", repo, flush=True)
    except Exception as error:
        print("FAIL", repo, type(error).__name__, str(error)[:200], flush=True)
try:
    snapshot_download("product-science/xlam-function-calling-60k-raw", repo_type="dataset",
                      local_dir="data/hf-campaign/xlam")
    print("OK xlam", flush=True)
except Exception as error:
    print("FAIL xlam", type(error).__name__, str(error)[:200], flush=True)
print("MORE_BASELINES_DONE", flush=True)
PY' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

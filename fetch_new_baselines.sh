#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export HF_ENDPOINT=http://127.0.0.1:8899 HF_HUB_DISABLE_XET=1 no_proxy=127.0.0.1 HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=120
for repo in EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m distilbert/distilgpt2 openai-community/gpt2 state-spaces/mamba-370m-hf state-spaces/mamba-790m-hf; do
  name=$(basename "$repo")
  [ -f "data/baselines/$name/.done" ] && continue
  .venv/bin/python - "$repo" "$name" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, name = sys.argv[1], sys.argv[2]
snapshot_download(repo, local_dir=f"data/baselines/{name}",
                  allow_patterns=["*.json","*.txt","*.safetensors"])
open(f"data/baselines/{name}/.done","w").write("ok")
print("DONE", name, flush=True)
PY
done
echo ALL-BASELINES-FETCHED

#!/usr/bin/env bash
# Does the agentic depth profile move as the donor scales? Same recipe, same corpus, same
# analysis, on the same family at 0.6B / 1.7B / 4B.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 \
       NO_PROXY=127.0.0.1,localhost HF_HUB_DISABLE_XET=1
STATUS=explog/SCALE_STATUS.txt

fetch() {  # <repo> <dir>
  .venv/bin/python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=f"data/baselines/{sys.argv[2]}",
                  allow_patterns=["*.safetensors*", "*.json", "*.jinja", "*token*", "*.txt"])
print("FETCHED", sys.argv[2])
PY
}

arm() {  # <tag> <repo> <dir> <batch>
  local tag="$1" repo="$2" dir="$3" batch="$4" start=$SECONDS
  [ -f "data/baselines/$dir/config.json" ] || fetch "$repo" "$dir" \
    > "explog/s_fetch_$tag.log" 2>&1 || { echo "$tag fetch rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/finetune_public.py --base "data/baselines/$dir" \
    --out "runs/lora/$tag" --batch-size "$batch" --max-length 768 \
    > "explog/s_ft_$tag.log" 2>&1 || { echo "$tag ft rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/python scripts/eval_suite.py --model "lora:data/baselines/$dir|runs/lora/$tag" \
    --rows 200 --device cuda --out "runs/evalsuite/ft-$tag.json" \
    > "explog/s_ev_$tag.log" 2>&1
  echo "$tag done rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

arm qwen3-17b Qwen/Qwen3-1.7B Qwen3-1.7B 4
arm qwen3-4b  Qwen/Qwen3-4B  Qwen3-4B  2
.venv/bin/python scripts/analyze_layers.py --tags qwen3-06b,qwen3-17b,qwen3-4b \
  --out runs/analysis/layer_profiles_scale.json > explog/layer_scale.log 2>&1
echo "scale-profiles rc=$?" >> "$STATUS"
echo SCALE_DONE >> "$STATUS"

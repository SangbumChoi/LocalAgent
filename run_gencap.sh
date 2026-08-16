#!/usr/bin/env bash
# Does the agent fine-tuning recipe cost the model its general language ability?
#
# Every open baseline is scored on six standard multiple-choice suites twice — the released
# weights, and the same weights with the LoRA adapter this report trains on the agent union
# corpus. Identical rows, identical scoring, so the difference is the recipe.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
export HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DISABLE_XET=1
STATUS=explog/GENCAP_STATUS.txt

.venv/bin/python scripts/fetch_corpora.py --plan general_corpora.json --out data/general \
  > explog/general_fetch.log 2>&1
echo "fetch rc=$?" >> "$STATUS"

pair() {  # <tag> <baseline-dir>
  local tag="$1" base="$2" start=$SECONDS
  .venv/bin/python scripts/gen_capability.py --model "data/baselines/$base" \
    --adapter "runs/lora/$tag" --tag "$tag" --rows 300 --device cuda \
    > "explog/gencap_$tag.log" 2>&1
  echo "$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
}

pair smollm2-135m     SmolLM2-135M-Instruct
pair lfm25-230m       LFM2.5-230M
pair lfm2-350m        LFM2-350M
pair granite-h-350m   granite-4.0-h-350m
pair granite-350m     granite-4.0-350m
pair smollm2-360m     SmolLM2-360M-Instruct
pair danube3-500m     h2o-danube3-500m-chat
pair qwen25-05b       Qwen2.5-0.5B-Instruct
pair qwen25-coder-05b Qwen2.5-Coder-0.5B-Instruct
pair lfm2-700m        LFM2-700M
pair qwen3-06b        Qwen3-0.6B
echo GENCAP_DONE >> "$STATUS"

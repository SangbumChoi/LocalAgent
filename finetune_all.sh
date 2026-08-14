#!/usr/bin/env bash
# One fine-tuning recipe, every public model, the same union corpus the local agents train on.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/FT_STATUS.txt
for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2-350M:lfm2-350m \
            SmolLM2-360M-Instruct:smollm2-360m h2o-danube3-500m-chat:danube3-500m \
            Qwen2.5-0.5B-Instruct:qwen25-05b Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b \
            LFM2-700M:lfm2-700m Qwen3-0.6B:qwen3-06b; do
  base="data/baselines/${pair%%:*}"; tag="${pair#*:}"; start=$SECONDS
  .venv/bin/python scripts/finetune_public.py --base "$base" --out "runs/lora/$tag" \
    --steps 600 > "explog/ft_$tag.log" 2>&1
  echo "ft-$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
  .venv/bin/python scripts/eval_suite.py --model "lora:$base|runs/lora/$tag" --rows 200 \
    --device cuda --out "runs/evalsuite/ft-$tag.json" > "explog/eval_ft_$tag.log" 2>&1
  echo "eval-ft-$tag rc=$?" >> "$STATUS"
done
echo FINETUNE_ALL_DONE >> "$STATUS"

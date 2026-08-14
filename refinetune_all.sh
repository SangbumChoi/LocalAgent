#!/usr/bin/env bash
# Re-run the whole fine-tuning block under the stabilised recipe, so every model in the table
# shares one code path rather than one that silently diverged on two of them.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/FT2_STATUS.txt
for _ in $(seq 1 120); do
  grep -q DONOR_SWEEP_DONE explog/DONOR_STATUS.txt 2>/dev/null && break
  sleep 60
done
for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2-350M:lfm2-350m \
            SmolLM2-360M-Instruct:smollm2-360m h2o-danube3-500m-chat:danube3-500m \
            Qwen2.5-0.5B-Instruct:qwen25-05b Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b \
            LFM2-700M:lfm2-700m Qwen3-0.6B:qwen3-06b; do
  base="data/baselines/${pair%%:*}"; tag="${pair#*:}"; start=$SECONDS
  .venv/bin/python scripts/finetune_public.py --base "$base" --out "runs/lora/$tag" \
    --steps 600 > "explog/ft2_$tag.log" 2>&1
  echo "ft-$tag rc=$? secs=$((SECONDS-start))" >> "$STATUS"
  .venv/bin/python scripts/eval_suite.py --model "lora:$base|runs/lora/$tag" --rows 200 \
    --device cuda --out "runs/evalsuite/ft-$tag.json" > "explog/eval_ft2_$tag.log" 2>&1
  echo "eval-ft-$tag rc=$?" >> "$STATUS"
done
for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2-350M:lfm2-350m \
            Qwen2.5-0.5B-Instruct:qwen25-05b; do
  base="data/baselines/${pair%%:*}"; tag="${pair#*:}"
  .venv/bin/python scripts/androidcontrol_clean.py --model "lora:$base|runs/lora/$tag" \
    --device cuda --out "runs/evalsuite-clean/ft-$tag.json" > "explog/clean_ft_$tag.log" 2>&1
  echo "clean-ft-$tag rc=$?" >> "$STATUS"
done
echo FINETUNE2_ALL_DONE >> "$STATUS"

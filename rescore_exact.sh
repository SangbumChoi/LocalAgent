#!/usr/bin/env bash
# Re-score the three coordinate/element suites under the corrected exact-match rule. Type match is
# unaffected; only step success changes, and it changes for every model equally.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/EXACT_STATUS.txt
one() { .venv/bin/python scripts/eval_suite.py --model "$2" --rows 200 --device cuda \
          --suites androidcontrol,mind2web,agentnet --out "runs/evalsuite/$1.json" \
          > "explog/ex_$1.log" 2>&1
        echo "$1 rc=$?" >> "$STATUS"; }
for size in 10m 96m; do one "catalog-$size" "catalog:runs/sft-catalog-$size/latest.pt"; done
one wide-10m catalog:runs/sft-wide-10m/latest.pt
one distill-10m catalog:runs/sft-distill2-10m/latest.pt
one distill-96m catalog:runs/sft-distill-96m/latest.pt
one profiled-10m catalog:runs/sft-profiled-10m/latest.pt
one profinv-10m catalog:runs/sft-profinv-10m/latest.pt
for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2.5-230M:lfm25-230m LFM2-350M:lfm2-350m \
            granite-4.0-h-350m:granite-h-350m granite-4.0-350m:granite-350m \
            SmolLM2-360M-Instruct:smollm2-360m h2o-danube3-500m-chat:danube3-500m \
            Qwen2.5-0.5B-Instruct:qwen25-05b Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b \
            LFM2-700M:lfm2-700m Qwen3-0.6B:qwen3-06b; do
  base="data/baselines/${pair%%:*}"; tag="${pair#*:}"
  one "$tag" "hf:$base"
  one "ft-$tag" "lora:$base|runs/lora/$tag"
done
echo EXACT_DONE >> "$STATUS"

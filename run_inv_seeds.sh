#!/usr/bin/env bash
# Two more seeds of the inverted-profile arm and one more of the uniform control: AgentNet moves
# 21.5 points between replicates, so a single pair cannot carry a claim this size.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/INVSEED_STATUS.txt
run() {  # <tag> <seed> <extra-flag>
  local tag="$1" seed="$2" flag="$3"
  sed -e "s|out_dir: runs/sft-distill2-10m|out_dir: runs/sft-$tag|" \
      -e "s|^  seed: 2026|  seed: $seed|" \
      configs/train/sft-distill2-10m.yaml > "configs/train/sft-$tag.yaml"
  .venv/bin/python scripts/profiled_sft.py --profile runs/analysis/lora_profile.json \
    --config "configs/train/sft-$tag.yaml" $flag > "explog/${tag}_sft.log" 2>&1
  echo "$tag sft rc=$?" >> "$STATUS"
  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-$tag/latest.pt" --rows 200 \
    --device cuda --out "runs/evalsuite/$tag.json" > "explog/${tag}_eval.log" 2>&1
  echo "$tag eval rc=$?" >> "$STATUS"
}
run invs101 101 --invert
run invs202 202 --invert
run unis101 101 ""
run unis202 202 ""
echo INVSEED_DONE >> "$STATUS"

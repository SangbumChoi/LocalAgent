#!/usr/bin/env bash
# The honest ≥90 campaign on face-h100: pool every downloaded corpus into one shard set and run
# the pretraining stage at 10x the token budget (16,000 steps ≈ 2.1B tokens), then the unchanged
# midtrain+SFT chain, isolating the pretraining-volume effect.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/BIGPRE_STATUS.txt

cat data/pretrain/ultrachat.txt data/pretrain/kimi-k3-distill.txt data/pretrain/hh-rlhf.txt \
    data/pretrain/qwen38-50k.txt data/pretrain/manus-distill.txt > data/pretrain/pool.txt
echo "pool $(du -sh data/pretrain/pool.txt | cut -f1) rc=$?" >> "$STATUS"
.venv/bin/python scripts/build_pretrain_variant.py --corpus data/pretrain/pool.txt \
  --reference data/shards/h100-mix --out data/shards/pt-big > explog/bp_pack.log 2>&1
echo "pack rc=$?" >> "$STATUS"

sed -e "s|shards_dir: data/shards/h100-mix|shards_dir: data/shards/pt-big|" \
    -e "s|total_steps: 1600|total_steps: 16000|" \
    -e "s|warmup_steps: 32|warmup_steps: 320|" \
    -e "s|ckpt_every: 400|ckpt_every: 2000|" \
    -e "s|eval_every: 400|eval_every: 2000|" \
    -e "s|out_dir: runs/ladder-96m-hybrid-seed2026|out_dir: runs/big-96m|" \
    configs/train/pretrain-ladder-96m-hybrid.yaml > configs/train/pretrain-big-96m.yaml
t0=$SECONDS
.venv/bin/localagent train pretrain configs/train/pretrain-big-96m.yaml > explog/bp_pre.log 2>&1 \
  || { echo "pretrain rc=$?" >> "$STATUS"; exit 1; }
echo "pretrain done secs=$((SECONDS-t0))" >> "$STATUS"

# Arm 2: big pretraining, unchanged wide+teacher SFT chain.
sed -e "s|init_from: .*|init_from: runs/big-96m/latest.pt|" \
    -e "s|out_dir: runs/midtrain-catalog-96m|out_dir: runs/midtrain-big-96m|" \
    configs/train/midtrain-catalog-96m.yaml > configs/train/midtrain-big-96m.yaml
sed -e "s|init_from: runs/midtrain-distill-96m/latest.pt|init_from: runs/midtrain-big-96m/latest.pt|" \
    -e "s|out_dir: runs/sft-distill-96m|out_dir: runs/sft-arm2-96m|" \
    configs/train/sft-distill-96m.yaml > configs/train/sft-arm2-96m.yaml
.venv/bin/localagent train midtrain configs/train/midtrain-big-96m.yaml > explog/bp_mid.log 2>&1 \
  || { echo "arm2 midtrain rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/localagent train sft configs/train/sft-arm2-96m.yaml > explog/bp_sft.log 2>&1 \
  || { echo "arm2 sft rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-arm2-96m/latest.pt --rows 200 \
  --device cuda --out runs/evalsuite/arm2-96m.json > explog/bp_eval.log 2>&1
echo "arm2 rc=$?" >> "$STATUS"
echo BIGPRE_DONE >> "$STATUS"

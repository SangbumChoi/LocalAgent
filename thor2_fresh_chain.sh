#!/usr/bin/env bash
[ -f runs/evalsuite/fresh-96m.json ] && { echo fresh chain already complete; exit 0; }
# Pack the pool with the ORIGINAL 16k tokenizer, then the fresh-token 10x arm: same 96M model,
# same 16,000 steps as arm 2, but over ~1B fresh pooled tokens instead of 4.5 epochs of the
# original corpus — the control that separates optimization from data novelty.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/FRESH_STATUS.txt

t0=$SECONDS
.venv/bin/python scripts/prepare_corpus.py \
  data/pretrain/ultrachat.docs.jsonl data/pretrain/kimi-k3-distill.docs.jsonl \
  data/pretrain/hh-rlhf.docs.jsonl data/pretrain/qwen38-50k.docs.jsonl \
  data/pretrain/manus-distill.docs.jsonl \
  --out data/shards/pool-16k --seq-len 2048 \
  --tokenizer bpe --tokenizer-path data/tokenizer-h100-16k.json --vocab-size 16384 \
  --reuse-tokenizer --staging-db /tmp/stage-16k.sqlite3 --no-near-dedup > explog/pack_pool16k.log 2>&1 \
  || { echo "pool-16k pack rc=$?" >> "$STATUS"; exit 1; }
echo "pool-16k packed secs=$((SECONDS-t0))" >> "$STATUS"

sed -e "s|shards_dir: data/shards/pt-big|shards_dir: data/shards/pool-16k|" \
    -e "s|out_dir: runs/big-96m|out_dir: runs/fresh-96m|" \
    configs/train/pretrain-big-96m.yaml > configs/train/pretrain-fresh-96m.yaml
.venv/bin/localagent train pretrain configs/train/pretrain-fresh-96m.yaml \
  > explog/fresh_pre.log 2>&1 || { echo "fresh pretrain rc=$?" >> "$STATUS"; exit 1; }
sed -e "s|init_from: .*|init_from: runs/fresh-96m/latest.pt|" \
    -e "s|out_dir: runs/midtrain-catalog-96m|out_dir: runs/midtrain-fresh-96m|" \
    configs/train/midtrain-catalog-96m.yaml > configs/train/midtrain-fresh-96m.yaml
sed -e "s|init_from: runs/midtrain-distill-96m/latest.pt|init_from: runs/midtrain-fresh-96m/latest.pt|" \
    -e "s|data/distill/wide.jsonl|data/distill2/train-clean.jsonl|" \
    -e "s|out_dir: runs/sft-distill-96m|out_dir: runs/sft-fresh-96m|" \
    configs/train/sft-distill-96m.yaml > configs/train/sft-fresh-96m.yaml
.venv/bin/localagent train midtrain configs/train/midtrain-fresh-96m.yaml > explog/fresh_mid.log 2>&1 \
  || { echo "fresh midtrain rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/localagent train sft configs/train/sft-fresh-96m.yaml > explog/fresh_sft.log 2>&1 \
  || { echo "fresh sft rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-fresh-96m/latest.pt --rows 200 \
  --device cuda --out runs/evalsuite/fresh-96m.json > explog/fresh_eval.log 2>&1
echo "fresh rc=$? DONE" >> "$STATUS"

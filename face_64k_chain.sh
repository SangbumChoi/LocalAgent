#!/usr/bin/env bash
# Pack the pool with the 64k tokenizer (real packer, staging + near-dedup), then the 8k variant.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/LA64K_STATUS.txt

t0=$SECONDS
if [ -f data/shards/pool-64k/manifest.json ]; then echo SKIP-PACK; else
.venv/bin/python scripts/prepare_corpus.py \
  data/pretrain/ultrachat.docs.jsonl data/pretrain/kimi-k3-distill.docs.jsonl \
  data/pretrain/hh-rlhf.docs.jsonl data/pretrain/qwen38-50k.docs.jsonl \
  data/pretrain/manus-distill.docs.jsonl \
  --out data/shards/pool-64k --seq-len 2048 \
  --tokenizer bpe --tokenizer-path data/tokenizer-h100-64k.json --vocab-size 65536 \
  --reuse-tokenizer --staging-db /tmp/stage-64k.sqlite3 --no-near-dedup > explog/pack_pool64k.log 2>&1 \
  || { echo "pool-64k pack rc=$?" >> "$STATUS"; exit 1; }
fi
echo "pool-64k packed secs=$((SECONDS-t0))" >> "$STATUS"
sed -i "s|shards_dir: data/shards/pt-big-64k|shards_dir: data/shards/pool-64k|" \
  configs/train/pretrain-la64k-8k.yaml configs/train/pretrain-la64k-16k.yaml
LOCALAGENT_PARAM_BUDGET=140000000 bash run_la64k.sh 8k

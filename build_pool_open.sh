#!/usr/bin/env bash
# Assemble the shipped gzips, pack pool-open with the pinned 16k tokenizer recipe, then hand
# the two arms to the daemons. Idempotent: pack() skips when the manifest exists.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src
for tag in toucan hermes glaive; do
  [ -f data/pretrain/open-$tag.docs.jsonl ] && continue
  cat /home/jovyan/sbchoi/incoming/open-$tag.docs.jsonl.gz.part* > /tmp/open-$tag.gz &&
    gunzip -c /tmp/open-$tag.gz > data/pretrain/open-$tag.docs.jsonl && rm -f /tmp/open-$tag.gz
done
wc -l data/pretrain/open-*.docs.jsonl >> explog/POOL_OPEN_STATUS.txt
if [ ! -f data/shards/pool-open/manifest.json ]; then
  .venv/bin/python scripts/prepare_corpus.py data/pretrain/open-toucan.docs.jsonl \
    data/pretrain/open-hermes.docs.jsonl data/pretrain/open-glaive.docs.jsonl \
    --out data/shards/pool-open --seq-len 2048 --tokenizer bpe \
    --tokenizer-path data/tokenizer-h100-16k.json --vocab-size 16384 --reuse-tokenizer \
    --staging-db /tmp/stage-pool-open.sqlite3 --no-near-dedup \
    > explog/pack_pool-open.log 2>&1 && echo "PACKED pool-open" >> explog/POOL_OPEN_STATUS.txt
fi
if [ -f data/shards/pool-open/manifest.json ]; then
  grep -q "skel-open-96m" experiments/queue-thor2.txt ||
    echo "bash skeleton_arm.sh skel-open-96m skeleton data/shards/pool-open" >> experiments/queue-thor2.txt
  grep -q "rand-open-96m" experiments/queue-face.txt ||
    echo "bash skeleton_arm.sh rand-open-96m random data/shards/pool-open" >> experiments/queue-face.txt
  echo "POOL-OPEN-ARMS-QUEUED" >> explog/POOL_OPEN_STATUS.txt
fi

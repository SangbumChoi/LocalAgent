#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src
pack() { # name, files...
  name=$1; shift
  [ -f data/shards/$name/manifest.json ] && { echo "SKIP $name"; return; }
  .venv/bin/python scripts/prepare_corpus.py "$@" --out data/shards/$name --seq-len 2048 \
    --tokenizer bpe --tokenizer-path data/tokenizer-h100-16k.json --vocab-size 16384 \
    --reuse-tokenizer --staging-db /tmp/stage-$name.sqlite3 --no-near-dedup \
    > explog/pack_$name.log 2>&1 && echo "PACKED $name" >> explog/R3_STATUS.txt
}
pack pool-distill data/pretrain/kimi-k3-distill.docs.jsonl data/pretrain/manus-distill.docs.jsonl data/pretrain/qwen38-50k.docs.jsonl
pack pool-chat data/pretrain/ultrachat.docs.jsonl data/pretrain/hh-rlhf.docs.jsonl
echo R3-PACKS-DONE >> explog/R3_STATUS.txt

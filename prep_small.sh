#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_corpus.py \
  data/raw/paper/download_state/001-cosmopedia_v2.jsonl \
  --out data/shards/h100-small --seq-len 2048 --tokenizer bpe --vocab-size 16384 \
  --tokenizer-path data/tokenizer-h100-small-16k.json > explog/prepare_small.log 2>&1
echo PREP_SMALL_RC=$? >> explog/prepare_small.log' </dev/null >/dev/null 2>&1 &
sleep 3; echo launched_small

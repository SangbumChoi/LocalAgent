#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_corpus.py \
  data/raw/paper/download_state/000-fineweb_edu_dedup.jsonl \
  data/raw/paper/download_state/001-cosmopedia_v2.jsonl \
  data/raw/paper/download_state/002-permissive_python.jsonl \
  --out data/shards/h100-mix --seq-len 2048 --tokenizer bpe --vocab-size 16384 \
  --tokenizer-path data/tokenizer-h100-16k.json > explog/prepare_corpus.log 2>&1
echo PREP_RC=$? >> explog/prepare_corpus.log' </dev/null >/dev/null 2>&1 &
sleep 5; tail -3 explog/prepare_corpus.log 2>/dev/null; echo launched

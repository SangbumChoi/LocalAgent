#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
ls data/raw/paper/ | head
echo "--- attempting the paper-tier corpus preparation (fail-closed gate expected) ---"
timeout 300 .venv/bin/python scripts/prepare_corpus.py data/raw/paper/mixture.jsonl \
  --source-manifest data/raw/paper/download_manifest.json \
  --out data/shards/paper-all --seq-len 2048 --tokenizer bpe --vocab-size 16384 \
  --tokenizer-path data/tokenizer-paper-16k.json 2>&1 | tail -25
echo "PAPER_PREP_RC=$?"

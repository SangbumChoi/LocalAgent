#!/usr/bin/env bash
# Download and characterise the five text pretraining corpora, then report tokens and wall clock.
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 \
       NO_PROXY=127.0.0.1,localhost HF_HUB_DISABLE_XET=1
.venv/bin/python scripts/fetch_corpora.py --plan pretrain_corpora.json \
  --out data/pretrain-src > explog/corpora_fetch.log 2>&1
echo "fetch rc=$?" >> explog/CORPORA_STATUS.txt
for tag in kimi-k3-distill qwen38-50k hh-rlhf ultrachat manus-distill; do
  .venv/bin/python scripts/prep_pretrain_corpus.py --source "data/pretrain-src/$tag" \
    --out "data/pretrain/$tag.txt" > "explog/corpus_prep_$tag.log" 2>&1
  echo "$tag rc=$?" >> explog/CORPORA_STATUS.txt
done
echo CORPORA_DONE >> explog/CORPORA_STATUS.txt' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

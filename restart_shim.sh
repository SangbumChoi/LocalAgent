#!/usr/bin/env bash
# The shim dies with the pod; the corpus fetch needs it back before anything can download.
cd /home/jovyan/sbchoi/localagent || exit 1
rm -rf hf_api_snapshot_v6 && tar xzf ../hf_snapshot6.tgz
pkill -f "hf[_]api_shim" 2>/dev/null; sleep 1
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot_v6 \
  --cache /home/jovyan/sbchoi/hfcache --port 8899 </dev/null > explog/shim6.log 2>&1 &
sleep 6
curl -s -o /dev/null -w "shim_probe=%{http_code}\n" http://127.0.0.1:8899/api/datasets/Anthropic/hh-rlhf
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 \
       NO_PROXY=127.0.0.1,localhost HF_HUB_DISABLE_XET=1
: > explog/CORPORA_STATUS.txt
.venv/bin/python scripts/fetch_corpora.py --plan pretrain_corpora.json \
  --out data/pretrain-src > explog/corpora_fetch.log 2>&1
echo "fetch rc=$?" >> explog/CORPORA_STATUS.txt
for tag in kimi-k3-distill qwen38-50k hh-rlhf ultrachat manus-distill; do
  .venv/bin/python scripts/prep_pretrain_corpus.py --source "data/pretrain-src/$tag" \
    --out "data/pretrain/$tag.txt" > "explog/corpus_prep_$tag.log" 2>&1
  echo "$tag rc=$?" >> explog/CORPORA_STATUS.txt
done
echo CORPORA_DONE >> explog/CORPORA_STATUS.txt' </dev/null >/dev/null 2>&1 &
echo RELAUNCHED

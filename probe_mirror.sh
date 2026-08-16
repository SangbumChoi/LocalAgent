#!/usr/bin/env bash
B=http://nexus.tossbank.bz/repository/huggingface-proxy
echo "--- codeparrot 246MB gz ---"
timeout 300 curl -s -o /tmp/p1 --noproxy '*' -L -w 'http=%{http_code} bytes=%{size_download} MBps=%{speed_download} t=%{time_total}\n' \
  "$B/datasets/codeparrot/codeparrot-clean/resolve/35a59fb025bc0a102f7d96eac09d145b896d487b/file-000000000001.json.gz" || echo "rc=$?"
echo "--- smollm 2.4GB parquet, first 200MB via range ---"
timeout 300 curl -s -o /tmp/p2 --noproxy '*' -L -r 0-209715199 -w 'http=%{http_code} bytes=%{size_download} MBps=%{speed_download} t=%{time_total}\n' \
  "$B/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/fineweb-edu-dedup/train-00000-of-00234.parquet" || echo "rc=$?"
ls -l /tmp/p1 /tmp/p2 2>/dev/null | awk '{print $9, $5}'

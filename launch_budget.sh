#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
for S in 2 6 10; do
setsid nohup bash -c "cd /home/jovyan/sbchoi/localagent
export PYTHONUNBUFFERED=1 PYTHONPATH=src
.venv/bin/python scripts/ablate_flywheel.py --rounds 5 --out runs/ablate/cuda-s0-x${S} --device cuda --seed 0 --sft-scale ${S} --freegen > explog/ablate_cuda-s0-x${S}.log 2>&1
echo \"ABL cuda-s0-x${S} rc=\$?\" >> explog/ABLATE_STATUS.txt" </dev/null >/dev/null 2>&1 &
done
sleep 3; pgrep -cf ablate_flywheel; echo budget_arms_launched
# characterise the mirror's large-file behaviour
echo "--- single large LFS file through the mirror ---"
timeout 180 curl -s -o /tmp/probe.parquet -w 'http=%{http_code} bytes=%{size_download} speed=%{speed_download} time=%{time_total}\n' \
  --noproxy '*' -L "http://nexus.tossbank.bz/repository/huggingface-proxy/datasets/codeparrot/codeparrot-clean/resolve/35a59fb025bc0a102f7d96eac09d145b896d487b/data/train-00000-of-00053.json.gz" || echo "curl_rc=$?"
ls -l /tmp/probe.parquet 2>/dev/null | awk '{print "downloaded_bytes="$5}'

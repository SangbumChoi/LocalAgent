#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
pkill -f "hf[_]api_shim" 2>/dev/null; sleep 1
mkdir -p /home/jovyan/sbchoi/hfcache
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot --cache /home/jovyan/sbchoi/hfcache --port 8899 </dev/null > explog/shim.log 2>&1 &
sleep 4; cat explog/shim.log
echo -n "api="; curl -s -o /dev/null -w '%{http_code}\n' -m 15 --noproxy '*' http://127.0.0.1:8899/api/datasets/Team-ACE/ToolACE
echo -n "small_file="; curl -s -o /dev/null -w '%{http_code} bytes=%{size_download}\n' -m 60 --noproxy '*' http://127.0.0.1:8899/datasets/Team-ACE/ToolACE/resolve/main/README.md
echo -n "range_of_large="; curl -s -o /dev/null -r 0-1048575 -w '%{http_code} bytes=%{size_download} t=%{time_total}\n' -m 900 --noproxy '*' "http://127.0.0.1:8899/datasets/codeparrot/codeparrot-clean/resolve/35a59fb025bc0a102f7d96eac09d145b896d487b/file-000000000001.json.gz"

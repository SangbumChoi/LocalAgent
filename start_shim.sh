#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
pkill -f "hf[_]api_shim" 2>/dev/null
sleep 1
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot --port 8899 </dev/null > explog/shim.log 2>&1 &
sleep 5
echo "--- shim log ---"; tail -3 explog/shim.log
echo -n "info="; curl -s -o /dev/null -w '%{http_code}\n' -m 10 --noproxy '*' http://127.0.0.1:8899/api/datasets/Team-ACE/ToolACE
echo -n "tree_bytes="; curl -s -m 10 --noproxy '*' 'http://127.0.0.1:8899/api/datasets/Team-ACE/ToolACE/tree/main' | wc -c
echo -n "resolve="; curl -s -o /dev/null -w '%{http_code} bytes=%{size_download}\n' -m 30 -L --noproxy '*' http://127.0.0.1:8899/datasets/Team-ACE/ToolACE/resolve/main/README.md

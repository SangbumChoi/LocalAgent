#!/usr/bin/env bash
# face-h100 half of the split: general-capability before/after, and the depth-band ablation.
cd /home/jovyan/sbchoi/localagent || exit 1
pkill -f "hf[_]api_shim" 2>/dev/null; sleep 1
setsid nohup .venv/bin/python hf_api_shim.py --snapshot hf_api_snapshot_v6 \
  --cache /home/jovyan/sbchoi/hfcache --port 8899 </dev/null > explog/shim_face.log 2>&1 &
sleep 6
curl -s -o /dev/null -w "shim=%{http_code}\n" http://127.0.0.1:8899/api/datasets/Anthropic/hh-rlhf
: > explog/GENCAP_STATUS.txt; : > explog/BAND_STATUS.txt
setsid nohup bash run_gencap.sh    </dev/null >/dev/null 2>&1 &
setsid nohup bash run_depth_band.sh </dev/null >/dev/null 2>&1 &
sleep 2; echo FACE_LAUNCHED

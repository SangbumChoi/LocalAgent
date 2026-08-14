#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
for size in 16m 35m 96m; do
setsid nohup bash -c "cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/localagent train midtrain configs/train/midtrain-catalog-${size}.yaml > explog/midtrain_catalog_${size}.log 2>&1
echo \"midtrain-catalog-${size} rc=\$?\" >> explog/LADDER_STATUS.txt
.venv/bin/localagent train sft configs/train/sft-catalog-${size}.yaml > explog/sft_catalog_${size}.log 2>&1
echo \"sft-catalog-${size} rc=\$?\" >> explog/LADDER_STATUS.txt" </dev/null >/dev/null 2>&1 &
done
sleep 25
for size in 10m 16m 35m 96m; do echo -n "$size: "; tail -1 explog/midtrain_catalog_${size}.log 2>/dev/null | cut -c1-95; echo; done

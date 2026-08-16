#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
: > explog/LADDER_STATUS.txt
for arm in 16m-hybrid 35m-hybrid 96m-hybrid; do
setsid nohup bash -c "cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/localagent train pretrain configs/train/pretrain-ladder-${arm}.yaml > explog/ladder_${arm}.log 2>&1
echo \"ladder-${arm} rc=\$?\" >> explog/LADDER_STATUS.txt" </dev/null >/dev/null 2>&1 &
done
sleep 8; pgrep -cf "localagent train"; echo ladder_launched

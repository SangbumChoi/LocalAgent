#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
for arm in attn hybrid; do
setsid nohup bash -c "cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/localagent train pretrain configs/train/pretrain-h100s-10m-${arm}.yaml > explog/train_10m_${arm}.log 2>&1
echo \"TRAIN ${arm} rc=\$?\" >> explog/TRAIN_STATUS.txt" </dev/null >/dev/null 2>&1 &
done
sleep 8; pgrep -cf "localagent train"; tail -3 explog/train_10m_attn.log 2>/dev/null

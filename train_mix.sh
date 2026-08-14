#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
: > explog/TRAIN_MIX_STATUS.txt
for arm in attn hybrid; do for seed in 2026 2027; do
setsid nohup bash -c "cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/localagent train pretrain configs/train/pretrain-mix-10m-${arm}-s${seed}.yaml > explog/train_mix_${arm}_s${seed}.log 2>&1
echo \"MIX ${arm} s${seed} rc=\$?\" >> explog/TRAIN_MIX_STATUS.txt" </dev/null >/dev/null 2>&1 &
done; done
sleep 10; pgrep -cf "localagent train"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

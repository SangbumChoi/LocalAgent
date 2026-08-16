#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash -c 'cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/finetune_public.py --base data/baselines/SmolLM2-135M-Instruct \
  --out runs/lora/smoke --steps 20 --rows 400 > explog/ft_smoke.log 2>&1
echo "smoke rc=$?" >> explog/FT_STATUS.txt' </dev/null >/dev/null 2>&1 &
echo LAUNCHED

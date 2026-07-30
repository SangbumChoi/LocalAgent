#!/usr/bin/env bash
# nanochat-style one-command, toy-scale training smoke. CPU-OK.
set -euo pipefail

echo "[speedrun] 1/6 install"          && pip install -e . >/dev/null
echo "[speedrun] 2/6 corpus"           && python scripts/prepare_corpus.py --sample \
  --out data/shards/sample --seq-len 128 --rows-per-shard 64 --val-fraction 0.1
echo "[speedrun] 3/6 pretrain"         && localagent train pretrain \
  configs/train/pretrain-speedrun.yaml
echo "[speedrun] 4/6 agent data"       && localagent synth configs/data/agent_synth.yaml
echo "[speedrun] 5/6 midtrain"         && localagent train midtrain \
  configs/train/midtrain-speedrun.yaml
echo "[speedrun] 6/6 SFT + RL + eval"  && python scripts/flywheel.py --quick --rounds 1
echo "[speedrun] done"

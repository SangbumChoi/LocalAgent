#!/usr/bin/env bash
# nanochat-style one-command, toy-scale, end-to-end run. CPU-OK.
# Each step is a no-op stub until its phase lands (see docs/ROADMAP.md).
set -euo pipefail

echo "[speedrun] 0/6  install"          && pip install -e . >/dev/null
echo "[speedrun] 1/6  tokenizer"        # TODO(phase-1): train BPE tokenizer
echo "[speedrun] 2/6  pretrain"         && localagent train pretrain configs/train/pretrain.yaml || true
echo "[speedrun] 3/6  synth agent data" && localagent synth configs/data/agent_synth.yaml || true
echo "[speedrun] 4/6  sft"              && localagent train sft configs/train/sft.yaml || true
echo "[speedrun] 5/6  eval"             && localagent eval runs/sft/latest.pt || true
echo "[speedrun] 6/6  export gguf"      && localagent export gguf runs/sft/latest.pt out/model.gguf || true
echo "[speedrun] done (stubs raise NotImplementedError until their phase is built)"

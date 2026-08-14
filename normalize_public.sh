#!/usr/bin/env bash
# Normalize every acquired public dataset into the repository's Conversation interchange.
# Each adapter is the repository's own, and every source is pinned by byte count and SHA-256.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=.venv/bin/python
HF=data/hf-campaign
OUT=data/public
mkdir -p "$OUT" explog
LOG=explog/normalize_public.log
: > "$LOG"

step() {
  local name="$1"; shift
  echo "=== $name ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "--- $name rc=$? ---" >> "$LOG"
}

# ToolACE — function calling
step toolace $PY scripts/normalize_toolace.py \
  --input "$HF/toolace/data.json" \
  --output-train "$OUT/toolace-train.jsonl" \
  --output-eval "$OUT/toolace-eval.jsonl" \
  --manifest "$OUT/toolace.manifest.json"

# AndroidControl — mobile UI control (public LLaMA-Factory-style mirror)
step androidcontrol_train $PY scripts/ingest_androidcontrol_json.py \
  --input "$HF/android_control_public_mirror/and_ctrl_train.json" \
  --output "$OUT/androidcontrol-train.jsonl" \
  --manifest "$OUT/androidcontrol-train.manifest.json" \
  --split train --source-revision 0248027f747c9d57bd09c14e8f044f9a8103dddd
step androidcontrol_test $PY scripts/ingest_androidcontrol_json.py \
  --input "$HF/android_control_public_mirror/and_ctrl_test.json" \
  --output "$OUT/androidcontrol-test.jsonl" \
  --manifest "$OUT/androidcontrol-test.manifest.json" \
  --split test --source-revision 0248027f747c9d57bd09c14e8f044f9a8103dddd

# AgentNet — desktop computer use. The repository's policy is evaluation-only, so only the
# eval output is used downstream.
AGENTNET_FILE=$(ls "$HF"/agentnet/*.jsonl 2>/dev/null | head -1)
step agentnet $PY scripts/ingest_agentnet_text.py \
  --input "$AGENTNET_FILE" \
  --train-output "$OUT/agentnet-unused-train.jsonl" \
  --eval-output "$OUT/agentnet-eval.jsonl" \
  --metadata-output "$OUT/agentnet.metadata.json" \
  --max-records 4000

# Mind2Web — browser control. The ingestion config is generated with real pins.
$PY - <<'PY' >> "$LOG" 2>&1
import hashlib
from pathlib import Path

shards = sorted(Path("data/hf-campaign/mind2web_train/data/train").glob("train_*.json"))[:3]
lines = [
    "schema_version: 1",
    "seed: 42",
    "enrichment_level: 3",
    "max_source_bytes: 8589934592",
    "outputs:",
    "  train: data/public/mind2web-train.jsonl",
    "manifest: data/public/mind2web.manifest.json",
    "",
    "sources:",
]
for index, shard in enumerate(shards):
    payload = shard.read_bytes()
    lines += [
        f"  - source_id: mind2web-train-{index}",
        "    dataset: osunlp/Mind2Web",
        "    subset: train",
        "    revision: 17ece8eb89862368edc0cc806acee6fca5163474",
        "    url: https://huggingface.co/datasets/osunlp/Mind2Web",
        "    license: cc-by-4.0",
        "    license_url: https://creativecommons.org/licenses/by/4.0/",
        "    adapter: mind2web_v1",
        "    split: train",
        f"    path: {shard}",
        f"    bytes: {len(payload)}",
        f"    sha256: {hashlib.sha256(payload).hexdigest()}",
        "    max_actions_per_record: 12",
    ]
Path("configs/data/mind2web-public-h100.yaml").write_text("\n".join(lines) + "\n")
print("wrote configs/data/mind2web-public-h100.yaml with", len(shards), "pinned shards")
PY

step mind2web $PY scripts/ingest_public_agent_data.py configs/data/mind2web-public-h100.yaml

echo "=== row counts ===" >> "$LOG"
for f in "$OUT"/*.jsonl; do echo "$(wc -l < "$f") $f" >> "$LOG"; done
echo NORMALIZE_DONE >> "$LOG"

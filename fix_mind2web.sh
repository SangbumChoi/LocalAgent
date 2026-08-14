#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
root = Path("/home/jovyan/sbchoi/localagent")
shards = sorted((root / "data/hf-campaign/mind2web_train/data/train").glob("train_*.json"))[:3]
lines = ["schema_version: 1", "seed: 42", "enrichment_level: 3",
         "max_source_bytes: 8589934592", "outputs:",
         f"  train: {root}/data/public/mind2web-train.jsonl",
         f"manifest: {root}/data/public/mind2web.manifest.json", "", "sources:"]
for index, shard in enumerate(shards):
    payload = shard.read_bytes()
    lines += [f"  - source_id: mind2web-train-{index}", "    dataset: osunlp/Mind2Web",
              "    subset: train", "    revision: 17ece8eb89862368edc0cc806acee6fca5163474",
              "    url: https://huggingface.co/datasets/osunlp/Mind2Web", "    license: cc-by-4.0",
              "    license_url: https://creativecommons.org/licenses/by/4.0/",
              "    adapter: mind2web_v1", "    split: train", f"    path: {shard}",
              f"    bytes: {len(payload)}", f"    sha256: {hashlib.sha256(payload).hexdigest()}",
              "    max_actions_per_record: 12"]
(root / "configs/data/mind2web-public-h100.yaml").write_text("\n".join(lines) + "\n")
print("pinned", len(shards), "shards")
PY
.venv/bin/python scripts/ingest_public_agent_data.py configs/data/mind2web-public-h100.yaml 2>&1 | tail -12
echo "MIND2WEB_RC=$?"
wc -l data/public/mind2web-train.jsonl 2>/dev/null

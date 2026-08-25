#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("/home/jovyan/sbchoi/localagent")
src = sorted((root / "data/hf-campaign/mind2web_train/data/train").glob("train_*.json"))[:11]
kept, dropped = [], 0
for shard in src:
    records = json.loads(shard.read_text())
    for record in records:
        actions = record.get("actions") or []
        ok = bool(actions) and all(
            isinstance(a.get("pos_candidates"), list) and a["pos_candidates"]
            and any(isinstance(c, dict) and isinstance(c.get("backend_node_id"), str)
                    and c["backend_node_id"] for c in a["pos_candidates"])
            for a in actions)
        (kept.append(record) if ok else None)
        dropped += 0 if ok else 1
    if len(kept) >= 5000:
        break
kept = kept[:5000]
out = root / "data/public/mind2web_grounded/train_0.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = json.dumps(kept).encode()
out.write_bytes(payload)
print(f"kept {len(kept)} fully grounded tasks, dropped {dropped} with an ungrounded action")

lines = ["schema_version: 1", "seed: 42", "enrichment_level: 3",
         "max_source_bytes: 8589934592", "outputs:",
         f"  train: {root}/data/public/mind2web-train.jsonl",
         f"manifest: {root}/data/public/mind2web.manifest.json", "", "sources:",
         "  - source_id: mind2web-train-grounded-subset", "    dataset: osunlp/Mind2Web",
         "    subset: train", "    revision: 17ece8eb89862368edc0cc806acee6fca5163474",
         "    url: https://huggingface.co/datasets/osunlp/Mind2Web", "    license: cc-by-4.0",
         "    license_url: https://creativecommons.org/licenses/by/4.0/",
         "    adapter: mind2web_v1", "    split: train", f"    path: {out}",
         f"    bytes: {len(payload)}", f"    sha256: {hashlib.sha256(payload).hexdigest()}",
         "    max_actions_per_record: 12"]
(root / "configs/data/mind2web-public-h100.yaml").write_text("\n".join(lines) + "\n")
PY
.venv/bin/python scripts/ingest_public_agent_data.py configs/data/mind2web-public-h100.yaml 2>&1 | tail -14
wc -l data/public/mind2web-train.jsonl 2>/dev/null

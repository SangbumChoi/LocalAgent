#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
from collections import Counter
from localagent.train.stage_data import read_conversations
for name in ("localagent-union-v1", "toolace-train", "mind2web-train", "androidcontrol-train", "agentnet-eval"):
    path = f"data/public/{name}.jsonl"
    try:
        rows = read_conversations(path)[:500]
    except Exception as error:
        print(name, "ERR", error); continue
    with_tools = sum(1 for r in rows if r.tools)
    sizes = Counter(len(r.tools or ()) for r in rows)
    print(f"{name:24s} rows={len(rows)} with_tools={with_tools} catalog_sizes={dict(list(sizes.items())[:5])}")
PY

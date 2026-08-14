#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_ENDPOINT=http://127.0.0.1:8899 NO_PROXY=127.0.0.1,localhost HF_HUB_DOWNLOAD_TIMEOUT=300
timeout 240 .venv/bin/python - <<'PY'
from datasets import load_dataset
import itertools, traceback
try:
    ds = load_dataset("HuggingFaceTB/smollm-corpus", name="fineweb-edu-dedup", split="train",
                      streaming=True, revision="3ba9d605774198c5868892d7a8deda78031a781f")
    rows = list(itertools.islice(iter(ds), 2))
    print("ROWS", len(rows), [list(r)[:4] for r in rows], len(rows[0]["text"]) if rows else 0)
except Exception:
    traceback.print_exc()
PY

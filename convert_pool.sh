#!/usr/bin/env bash
# The packer reads a .txt as one document; our pool is one document per line. Re-express as JSONL.
cd /home/jovyan/sbchoi/localagent || exit 1
.venv/bin/python - <<'PY'
import json
from pathlib import Path

for tag in ("ultrachat", "kimi-k3-distill", "hh-rlhf", "qwen38-50k", "manus-distill"):
    src = Path(f"data/pretrain/{tag}.txt")
    dst = Path(f"data/pretrain/{tag}.docs.jsonl")
    n = 0
    with src.open(encoding="utf-8", errors="replace") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            line = line.strip()
            if len(line) >= 200:
                out.write(json.dumps({"text": line, "meta": {"mixture_source": tag}},
                                     ensure_ascii=False) + "\n")
                n += 1
    print(tag, n)
print("CONVERT_DONE")
PY

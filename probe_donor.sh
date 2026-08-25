#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
import json
from collections import defaultdict
from safetensors.torch import load_file
sd = load_file("data/hf-campaign/localagent_tiny_model/model.safetensors")
groups = defaultdict(lambda: [0, 0])
for k, v in sd.items():
    key = k.split(".")[0] if not k.startswith("blocks.") else "blocks." + ".".join(k.split(".")[2:3])
    groups[key][0] += 1
    groups[key][1] += v.numel()
print("total tensors", len(sd), "params", sum(v.numel() for v in sd.values()))
for k, (n, p) in sorted(groups.items()):
    print(f"  {k:24s} tensors={n:3d} params={p:,}")
print("\nfirst 12 keys:", list(sd)[:12])
heads = json.load(open("data/hf-campaign/localagent_tiny_model/heads.json"))
print("\nheads.json keys:", list(heads)[:8])
for k in list(heads)[:6]:
    v = heads[k]
    print("  ", k, type(v).__name__, (len(v) if isinstance(v, list) else str(v)[:60]))
PY

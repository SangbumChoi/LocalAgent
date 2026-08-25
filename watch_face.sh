#!/usr/bin/env bash
# face-h100 chain: wait for the band arms and the catalog ToolBench pass to release the GPU, then
# backfill the seed replicates' missing suites (their reports predate the 6/7-suite harness, and
# the ledger's noise floor needs measured spreads for bfcl and toolbench, not placeholders), then
# run the pretraining-corpus comparison on an otherwise idle box so its wall clock is honest.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
while true; do
  grep -q BAND_DONE explog/BAND_STATUS.txt 2>/dev/null \
    && grep -q CATALOG_ARMS_DONE explog/TOOLBENCH_STATUS.txt 2>/dev/null && break
  sleep 120
done
.venv/bin/python - <<'PY' > explog/seed_backfill.log 2>&1
import json, subprocess
from pathlib import Path

for tag in ("seed-101", "seed-202", "seed-303", "repeat-1", "repeat-2", "repeat-3"):
    path = Path(f"runs/evalsuite/{tag}.json")
    if not path.exists():
        continue
    reported = json.loads(path.read_text())
    missing = [s for s in ("bfcl", "toolbench") if s not in reported.get("suites", {})]
    if not missing:
        continue
    spec = f"{reported['kind']}:{reported['location']}"
    subprocess.run([".venv/bin/python", "scripts/eval_suite.py", "--model", spec,
                    "--rows", "200", "--device", "cuda", "--suites", ",".join(missing),
                    "--out", str(path)], check=False)
    print(tag, "backfilled", missing, flush=True)
print("SEED_BACKFILL_DONE", flush=True)
PY
echo SEEDS_BACKFILLED >> explog/PRECORP_STATUS.txt
bash run_pretrain_corpora.sh

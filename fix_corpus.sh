#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
pkill -f "download_pretrain_mixture" 2>/dev/null; sleep 2
cd data/raw/paper/download_state
python3 - <<'PY'
import json, os
for name in sorted(os.listdir('.')):
    if not name.endswith('.manifest.json'): continue
    stem = name[:-len('.manifest.json')]
    data = stem + '.jsonl'
    m = json.load(open(name))
    want = m.get('bytes') or m.get('data_bytes') or m.get('size_bytes')
    have = os.path.getsize(data) if os.path.exists(data) else None
    print(stem, 'manifest_bytes=', want, 'actual=', have, 'OK' if want == have else 'MISMATCH')
    if want != have:
        for f in (data, name):
            if os.path.exists(f): os.remove(f); print('  removed', f)
PY
cd /home/jovyan/sbchoi/localagent
rm -f data/raw/paper/download_state/*.tmp
ls -la data/raw/paper/download_state | tail -8

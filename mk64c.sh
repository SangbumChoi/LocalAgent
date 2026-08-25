cd /home/jovyan/sbchoi/localagent
python3 - <<'PY'
import json
m = json.load(open("data/shards/pt-big-64k/manifest.json"))
for key in ("splits", "corpus_audit", "corpus_variant"):
    m.pop(key, None)
json.dump(m, open("data/shards/pt-big-64k/manifest.json", "w"), indent=1)
print("keys now:", sorted(m.keys()))
PY
sed -n '3425,3450p' src/localagent/data/pretrain_corpus.py

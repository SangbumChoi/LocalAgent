cd /home/jovyan/sbchoi/localagent
grep -rn "packed corpus vocabulary" src/ --include=*.py
python3 - <<'PY'
import json
d = json.load(open("data/shards/pt-big/manifest.json"))
print({k: v for k, v in d.items() if k not in ("corpus_audit", "corpus_variant")})
PY
ls data/shards/pt-big/

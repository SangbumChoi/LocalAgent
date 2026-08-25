cd /home/jovyan/sbchoi/localagent
echo "=== h100-mix generations ==="; du -s data/shards/h100-mix/generations/*/ 2>/dev/null | head -4
echo "=== pt-big generations ==="; du -s data/shards/pt-big/generations/*/ 2>/dev/null | head -6
echo "=== pt-ultrachat generations ==="; du -s data/shards/pt-ultrachat/generations/*/ 2>/dev/null | head -4
echo "=== dir name comparison ==="
ls data/shards/h100-mix/generations/ | sort > /tmp/g1; ls data/shards/pt-big/generations/ | sort > /tmp/g2
comm -12 /tmp/g1 /tmp/g2 | wc -l; comm -13 /tmp/g1 /tmp/g2 | wc -l
echo "=== manifest splits token totals ==="
python3 - <<'PY'
import json
for tag in ("h100-mix", "pt-big", "pt-ultrachat"):
    try:
        m = json.load(open(f"data/shards/{tag}/manifest.json"))
        tt = m.get("total_tokens"); v = m.get("vocab_size")
        gens = {s: [e.get("generation") for e in sp.get("shards", [])][:1]
                for s, sp in (m.get("splits") or {}).items()}
        print(tag, "total_tokens=", tt, "vocab=", v, "gen-refs=", gens)
    except Exception as e:
        print(tag, "ERR", e)
PY

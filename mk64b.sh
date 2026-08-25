cd /home/jovyan/sbchoi/localagent
python3 - <<'PY'
import json
m = json.load(open("data/shards/pt-big/manifest.json"))
print("top-level keys:", sorted(m.keys()))
PY
rm -rf data/shards/pt-big-64k
mkdir -p data/shards/pt-big-64k/generations
ln data/shards/pt-big/filtered.jsonl data/shards/pt-big-64k/filtered.jsonl
python3 - <<'PY'
import hashlib, json
m = json.load(open("data/shards/pt-big/manifest.json"))
# Drop every key that records a packed-token artifact; the trainer repacks for the new tokenizer.
clean = {k: v for k, v in m.items()
         if k in ("version", "val_fraction", "total_documents", "corpus_audit", "corpus_variant",
                  "token_dtype", "splits", "sources")}
clean["version"] = m.get("version", 2)
sha = hashlib.sha256(open("data/tokenizer-h100-64k.json", "rb").read()).hexdigest()
clean["vocab_size"] = 65536
clean["tokenizer_training"] = {"path": "data/tokenizer-h100-64k.json", "vocab_size": 65536,
                               "requested_vocab_size": 65536, "kind": "bpe", "trained": True,
                               "artifact": {"path": "data/tokenizer-h100-64k.json",
                                            "sha256": sha}}
json.dump(clean, open("data/shards/pt-big-64k/manifest.json", "w"), indent=1)
print("clean manifest keys:", sorted(clean.keys()))
PY
echo REBUILT

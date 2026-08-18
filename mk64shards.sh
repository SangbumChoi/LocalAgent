cd /home/jovyan/sbchoi/localagent
mkdir -p data/shards/pt-big-64k
ln -sf ../pt-big/filtered.jsonl data/shards/pt-big-64k/filtered.jsonl
[ -e data/shards/pt-big-64k/generations ] || ln -s ../pt-big/generations data/shards/pt-big-64k/generations
python3 - <<'PY'
import hashlib, json
m = json.load(open("data/shards/pt-big/manifest.json"))
sha = hashlib.sha256(open("data/tokenizer-h100-64k.json", "rb").read()).hexdigest()
m["vocab_size"] = 65536
t = m.get("tokenizer_training", {})
t.update({"path": "data/tokenizer-h100-64k.json", "vocab_size": 65536,
          "requested_vocab_size": 65536,
          "artifact": {"path": "data/tokenizer-h100-64k.json", "sha256": sha,
                       "bytes": len(open("data/tokenizer-h100-64k.json","rb").read())},
          "retokenized_from": "pt-big (16k); text shards shared, token counts stale"})
m["tokenizer_training"] = t
json.dump(m, open("data/shards/pt-big-64k/manifest.json", "w"), indent=1)
print("manifest written")
PY
sed -i "s|shards_dir: data/shards/pt-big|shards_dir: data/shards/pt-big-64k|" \
  configs/train/pretrain-la64k-8k.yaml configs/train/pretrain-la64k-16k.yaml
echo SHARDS64_READY

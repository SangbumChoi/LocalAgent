cd /home/jovyan/sbchoi/localagent
sed -n '3162,3210p' src/localagent/data/pretrain_corpus.py
grep -rn "pack_disk_backed_shards" src/ scripts/ --include=*.py | grep -v "def pack_disk" | head -5

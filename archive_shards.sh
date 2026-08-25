#!/usr/bin/env bash
# Permanence pass for a finished shard dir: verify, checksum into the registry, protect the
# staging DB, and drop a restorable tarball of the packed tokens in the archive folder.
#   bash archive_shards.sh data/shards/pool-16k
cd /home/jovyan/sbchoi/localagent || exit 1
dir="$1"; name=$(basename "$dir")
[ -f "$dir/manifest.json" ] || { echo "no manifest in $dir — not finished"; exit 1; }
mkdir -p /home/jovyan/sbchoi/archive experiments
sha=$(sha256sum "$dir/manifest.json" | cut -d' ' -f1)
size=$(du -sh "$dir" | cut -f1)
chmod -w "$dir/manifest.json" "$dir"/corpus-staging.sqlite3 2>/dev/null
tar czf "/home/jovyan/sbchoi/archive/${name}-packed.tgz" -C "$dir" manifest.json generations 2>/dev/null
echo "$(date -u +%FT%TZ) ARCHIVED $name size=$size manifest_sha=$sha tar=$(du -sh /home/jovyan/sbchoi/archive/${name}-packed.tgz | cut -f1)" >> experiments/registry.log
tail -1 experiments/registry.log

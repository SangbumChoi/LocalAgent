#!/usr/bin/env bash
# xLAM's own train shards: the one source that adds genuinely new tool diversity, which is what
# the student lacks. Test shards stay out.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
INPUTS=""
for f in data/hf-campaign/xlam/data/train/*.parquet; do INPUTS="$INPUTS --input $f"; done
.venv/bin/python scripts/normalize_xlam_parquet.py $INPUTS --skip-invalid --split train \
  --max-records 24000 --output data/public/xlam-train.jsonl \
  --manifest data/public/xlam-train.manifest.json > explog/xlam_train.log 2>&1
echo "xlam-train rc=$?" >> explog/XLAM_TRAIN_STATUS.txt
wc -l data/public/xlam-train.jsonl >> explog/XLAM_TRAIN_STATUS.txt
echo XLAM_TRAIN_DONE >> explog/XLAM_TRAIN_STATUS.txt

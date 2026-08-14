#!/usr/bin/env bash
# Normalize xLAM's official test shards into an eval suite. A second function-calling surface with
# a split the dataset authors drew, rather than one we drew ourselves.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/pip install -q pyarrow >> explog/xlam.log 2>&1
.venv/bin/python scripts/normalize_xlam_parquet.py \
  --input data/hf-campaign/xlam/data/test/shard_0.parquet \
  --input data/hf-campaign/xlam/data/test/shard_1.parquet \
  --input data/hf-campaign/xlam/data/test/shard_2.parquet \
  --skip-invalid --split eval --output data/public/xlam-test.jsonl \
  --manifest data/public/xlam-test.manifest.json >> explog/xlam.log 2>&1
echo "xlam rc=$?" >> explog/xlam.log
wc -l data/public/xlam-test.jsonl >> explog/xlam.log 2>&1
echo XLAM_DONE >> explog/xlam.log

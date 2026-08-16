#!/usr/bin/env bash
# Does the pretraining corpus decide how good the agent gets after post-training?
#
# Five open corpora, each packed into the reference shard layout and run through the identical
# chain at an identical step budget, so the corpus is the only variable and the token budget
# consumed is the same for all of them. Wall clock is recorded per stage.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
STATUS=explog/PRECORP_STATUS.txt

arm() {  # <tag>
  local tag="$1" name="pc-$1" start=$SECONDS
  .venv/bin/python scripts/build_pretrain_variant.py --corpus "data/pretrain/$tag.txt" \
    --reference data/shards/h100-mix --out "data/shards/pt-$tag" \
    > "explog/p_pack_$tag.log" 2>&1 || { echo "$tag pack rc=$?" >> "$STATUS"; return 1; }
  local packed=$((SECONDS-start))

  sed -e "s|shards_dir: data/shards/h100-mix|shards_dir: data/shards/pt-$tag|" \
      -e "s|out_dir: runs/ladder-96m-hybrid-seed2026|out_dir: runs/pretrain-$name|" \
      configs/train/pretrain-ladder-96m-hybrid.yaml > "configs/train/pretrain-$name.yaml"
  local t0=$SECONDS
  .venv/bin/localagent train pretrain "configs/train/pretrain-$name.yaml" \
    > "explog/p_pre_$tag.log" 2>&1 || { echo "$tag pretrain rc=$?" >> "$STATUS"; return 1; }
  local pretrained=$((SECONDS-t0))

  sed -e "s|init_from: .*|init_from: runs/pretrain-$name/latest.pt|" \
      -e "s|out_dir: runs/midtrain-catalog-96m|out_dir: runs/midtrain-$name|" \
      configs/train/midtrain-catalog-96m.yaml > "configs/train/midtrain-$name.yaml"
  sed -e "s|init_from: runs/midtrain-catalog-96m/latest.pt|init_from: runs/midtrain-$name/latest.pt|" \
      -e "s|out_dir: runs/sft-catalog-96m|out_dir: runs/sft-$name|" \
      configs/train/sft-catalog-96m.yaml > "configs/train/sft-$name.yaml"
  t0=$SECONDS
  .venv/bin/localagent train midtrain "configs/train/midtrain-$name.yaml" \
    > "explog/p_mid_$tag.log" 2>&1 || { echo "$tag midtrain rc=$?" >> "$STATUS"; return 1; }
  .venv/bin/localagent train sft "configs/train/sft-$name.yaml" \
    > "explog/p_sft_$tag.log" 2>&1 || { echo "$tag sft rc=$?" >> "$STATUS"; return 1; }
  local posted=$((SECONDS-t0))

  .venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-$name/latest.pt" --rows 200 \
    --device cuda --out "runs/evalsuite/$name.json" > "explog/p_eval_$tag.log" 2>&1
  echo "$tag done rc=$? pack=${packed}s pretrain=${pretrained}s posttrain=${posted}s total=$((SECONDS-start))s" \
    >> "$STATUS"
}

for tag in qwen38-50k manus-distill hh-rlhf kimi-k3-distill ultrachat; do
  arm "$tag"
done
echo PRECORP_DONE >> "$STATUS"

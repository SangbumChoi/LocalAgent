#!/usr/bin/env bash
# One Solar-regime arm: project a donor REGION into the 96M student, then run the FULL fresh-pool
# pretraining (16,000 steps) and the standard midtrain+SFT chain. The random-init control is the
# fresh-96m arm on the identical corpus and budget.
#   bash skeleton_arm.sh <name> <region>     e.g.  skeleton_arm.sh skel-96m skeleton
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 LOCALAGENT_RESUME_LINEAGE=warn
# face's older torch (2.6.0a0) needs numpy 1.26 while the shared venv carries 2.5.2 for thor2;
# a pod-local copy shadows it on face only.
case "$(hostname)" in face-*)
  [ -d /tmp/np126 ] || .venv/bin/pip install -q --target=/tmp/np126 'numpy==1.26.4'
  export PYTHONPATH="/tmp/np126:$PYTHONPATH";;
esac
name="$1"; region="$2"; STATUS=explog/SKEL_STATUS.txt; t0=$SECONDS
shards="${3:-data/shards/pt-big}"; seed="${4:-}"; steps="${5:-}"  # arm2's corpus by default: its random-init control exists

if [ "$region" != random ] && [ -f "runs/donor-init/$name/latest.pt" ]; then
  echo "$name INIT-SKIP (artifact exists)" >> "$STATUS"
elif [ "$region" != random ]; then   # region-random: the chain-matched random-init control
.venv/bin/python scripts/cross_donor_init.py --donor data/baselines/Qwen3-0.6B \
  --student runs/ladder-96m-hybrid-seed2026/latest.pt --region "$region" \
  --out "runs/donor-init/$name/latest.pt" > "explog/sk_init_$name.log" 2>&1 \
  || { echo "$name init rc=$?" >> "$STATUS"; exit 1; }
fi

sed -e "s|shards_dir: data/shards/pt-big|shards_dir: ${shards}|" \
    -e "s|out_dir: runs/big-96m|out_dir: runs/pre-$name|" \
    configs/train/pretrain-big-96m.yaml > "configs/train/pretrain-$name.yaml"
SKEL_REGION="$region" SKEL_SEED="$seed" SKEL_STEPS="$steps" python3 - "$name" <<'PY'
import sys, yaml
name = sys.argv[1]
path = f"configs/train/pretrain-{name}.yaml"
d = yaml.safe_load(open(path))
import os
if os.environ.get('SKEL_REGION') == 'random':
    d.pop('init_from', None)
else:
    d["init_from"] = f"runs/donor-init/{name}/latest.pt"
if os.environ.get('SKEL_SEED'):
    d['seed'] = int(os.environ['SKEL_SEED'])
if os.environ.get('SKEL_STEPS'):
    d.setdefault('schedule', {})['total_steps'] = int(os.environ['SKEL_STEPS'])
yaml.safe_dump(d, open(path, "w"), sort_keys=False)
PY
if .venv/bin/python -c "import json,sys; m=json.load(open('runs/pre-$name/metrics.json')); sys.exit(0 if m.get('steps_completed',0)>=${steps:-16000} else 1)" 2>/dev/null; then
  echo "$name PRETRAIN-SKIP (already complete)" >> "$STATUS"
else
  .venv/bin/localagent train pretrain "configs/train/pretrain-$name.yaml" \
    > "explog/sk_pre_$name.log" 2>&1 || { echo "$name pretrain rc=$?" >> "$STATUS"; exit 1; }
fi

sed -e "s|init_from: .*|init_from: runs/pre-$name/latest.pt|" \
    -e "s|out_dir: runs/midtrain-catalog-96m|out_dir: runs/mid-$name|" \
    configs/train/midtrain-catalog-96m.yaml > "configs/train/midtrain-$name.yaml"
sed -e "s|init_from: runs/midtrain-distill-96m/latest.pt|init_from: runs/mid-$name/latest.pt|" \
    -e "s|data/distill/wide.jsonl|data/distill2/train-clean.jsonl|" \
    -e "s|out_dir: runs/sft-distill-96m|out_dir: runs/sft-$name|" \
    configs/train/sft-distill-96m.yaml > "configs/train/sft-$name.yaml"
.venv/bin/localagent train midtrain "configs/train/midtrain-$name.yaml" > "explog/sk_mid_$name.log" 2>&1 \
  || { echo "$name midtrain rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/localagent train sft "configs/train/sft-$name.yaml" > "explog/sk_sft_$name.log" 2>&1 \
  || { echo "$name sft rc=$?" >> "$STATUS"; exit 1; }
.venv/bin/python scripts/eval_suite.py --model "catalog:runs/sft-$name/latest.pt" --rows 200 \
  --device cuda --out "runs/evalsuite/$name.json" > "explog/sk_eval_$name.log" 2>&1
echo "$name done rc=$? secs=$((SECONDS-t0))" >> "$STATUS"

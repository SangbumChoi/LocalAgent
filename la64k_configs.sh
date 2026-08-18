#!/usr/bin/env bash
# The 64k-vocabulary long-context variants: same 18-layer hybrid backbone, LFM-class vocabulary,
# RoPE theta raised to 1e6 for the longer windows. Parameter count rises with the embedding and
# is reported, not hidden.
cd /home/jovyan/sbchoi/localagent || exit 1

for pair in "8k:8192" "16k:16384"; do
  tag="${pair%%:*}"; ctx="${pair##*:}"
  sed -e "s|name: webgpu-96m-hybrid|name: la64k-${tag}|" \
      -e "s|vocab_size: 16384|vocab_size: 65536|" \
      -e "s|max_seq_len: 4096|max_seq_len: ${ctx}|" \
      -e "s|rope_theta: 10000.0|rope_theta: 1000000.0|" \
      configs/model/webgpu-96m-hybrid.yaml > "configs/model/la64k-${tag}.yaml"

  sed -e "s|model_config: configs/model/webgpu-96m-hybrid.yaml|model_config: configs/model/la64k-${tag}.yaml|" \
      -e "s|path: data/tokenizer-h100-16k.json|path: data/tokenizer-h100-64k.json|" \
      -e "s|shards_dir: data/shards/h100-mix|shards_dir: data/shards/pt-big|" \
      -e "s|total_steps: 1600|total_steps: 16000|" \
      -e "s|warmup_steps: 32|warmup_steps: 320|" \
      -e "s|ckpt_every: 400|ckpt_every: 2000|" \
      -e "s|eval_every: 400|eval_every: 2000|" \
      -e "s|out_dir: runs/ladder-96m-hybrid-seed2026|out_dir: runs/pre-la64k-${tag}|" \
      configs/train/pretrain-ladder-96m-hybrid.yaml > "configs/train/pretrain-la64k-${tag}.yaml"

  sed -e "s|model_config: configs/model/webgpu-96m-hybrid.yaml|model_config: configs/model/la64k-${tag}.yaml|" \
      -e "s|path: data/tokenizer-h100-16k.json|path: data/tokenizer-h100-64k.json|" \
      -e "s|init_from: .*|init_from: runs/pre-la64k-${tag}/latest.pt|" \
      -e "s|out_dir: runs/midtrain-catalog-96m|out_dir: runs/mid-la64k-${tag}|" \
      configs/train/midtrain-catalog-96m.yaml > "configs/train/midtrain-la64k-${tag}.yaml"

  sed -e "s|model_config: configs/model/webgpu-96m-hybrid.yaml|model_config: configs/model/la64k-${tag}.yaml|" \
      -e "s|path: data/tokenizer-h100-16k.json|path: data/tokenizer-h100-64k.json|" \
      -e "s|init_from: runs/midtrain-distill-96m/latest.pt|init_from: runs/mid-la64k-${tag}/latest.pt|" \
      -e "s|data/distill/wide.jsonl|data/distill2/train-clean.jsonl|" \
      -e "s|out_dir: runs/sft-distill-96m|out_dir: runs/sft-la64k-${tag}|" \
      configs/train/sft-distill-96m.yaml > "configs/train/sft-la64k-${tag}.yaml"
done
echo CONFIGS_DONE

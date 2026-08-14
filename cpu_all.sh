#!/usr/bin/env bash
# CPU decode cost for every public model plus our own, four threads each, run several at a time on
# the host's spare cores so the GPU sweep is untouched.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
one() {
  local tag="$1" spec="$2"
  .venv/bin/python scripts/eval_suite.py --model "$spec" --rows 40 --device cpu \
    --out "runs/evalsuite-cpu/$tag.json" > "explog/cpu_$tag.log" 2>&1
  echo "cpu-$tag rc=$?" >> explog/CPU_STATUS.txt
}
for pair in SmolLM2-135M-Instruct:smollm2-135m LFM2-350M:lfm2-350m \
            SmolLM2-360M-Instruct:smollm2-360m Qwen2.5-0.5B-Instruct:qwen25-05b \
            Qwen2.5-Coder-0.5B-Instruct:qwen25-coder-05b h2o-danube3-500m-chat:danube3-500m \
            LFM2-700M:lfm2-700m Qwen3-0.6B:qwen3-06b; do
  one "${pair#*:}" "hf:data/baselines/${pair%%:*}" &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
one catalog-10m catalog:runs/sft-catalog-10m/latest.pt &
wait
echo CPU_ALL_DONE >> explog/CPU_STATUS.txt

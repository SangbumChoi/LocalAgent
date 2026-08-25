#!/usr/bin/env bash
# CPU-only decode budget for the motivation: what a machine without WebGPU can actually run.
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for threads in 1 4; do
  OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
  .venv/bin/python - <<PY >> explog/cpu_bench.log 2>&1
import json, time, torch
torch.set_num_threads($threads)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.inference.generate import generate

rows = []
for name in ("ultra-tiny-1m", "tiny-30m-byte", "small-90m"):
    try:
        cfg = ModelConfig.from_yaml(f"configs/model/{name}.yaml")
    except FileNotFoundError:
        continue
    model = LocalAgentLM(cfg).eval()
    tok = load_tokenizer("byte")
    ids = torch.tensor([tok.encode("<|user|>Click 'Save changes' at the bottom.<|assistant|>")])
    def forward(batch):
        out = model(batch)
        return out[0] if isinstance(out, tuple) else out

    with torch.no_grad():
        forward(ids)                                  # warm-up
        start = time.perf_counter()
        forward(ids)
        prefill = time.perf_counter() - start
        start = time.perf_counter()
        steps = 32
        sequence = ids
        for _ in range(steps):
            logits = forward(sequence)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            sequence = torch.cat([sequence, nxt], dim=1)
        decode = time.perf_counter() - start
    rows.append({"model": name, "threads": $threads, "params_m": round(model.num_params()/1e6, 3),
                 "prefill_ms": round(prefill*1000, 2), "decode_tok_s": round(steps/decode, 1),
                 "ms_per_token": round(decode/steps*1000, 2),
                 "weights_mb_fp32": round(model.num_params()*4/1e6, 1)})
    print(json.dumps(rows[-1]), flush=True)
PY
done
echo CPU_BENCH_DONE >> explog/cpu_bench.log

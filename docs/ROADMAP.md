# Roadmap

Phased build order. Each phase is independently runnable and leaves the repo in a working state.
Stubs reference these phase numbers in their `TODO(phase-N)` markers.

## Phase 0 — Scaffold (this PR)
- [x] Research synthesis ([RESEARCH.md](./RESEARCH.md)) and architecture ([ARCHITECTURE.md](./ARCHITECTURE.md)).
- [x] Repo skeleton: package layout, configs, CLI, demos, tests, `pyproject.toml`.
- [x] Canonical `Conversation` schema and tool registry interface.
- [x] Model config + param-budget assertion; tiny Llama-style block implemented.
- Outcome: `pip install -e .`, `localagent --help`, `pytest` (schema/model/config tests) pass.

## Phase 1 — Model & tokenizer
- [ ] Finish the decoder (GQA + RoPE + SwiGLU + KV cache), forward/generate.
- [ ] Train BPE tokenizer with agent special tokens; round-trip tests.
- [ ] `tiny-30m` constructs and runs a forward pass on CPU.
- Exit: untrained model generates tokens; param count < 100M asserted.

## Phase 2 — Pretrain
- [ ] Corpus download + quality filter + shard packing.
- [ ] `pretrain.py` loop (AdamW/Muon, cosine LR, grad accum, ckpt/resume) on CPU & GPU.
- [ ] `speedrun.sh` pretrains a toy model end-to-end.
- Exit: loss decreases; coherent-ish completions at toy scale.

## Phase 3 — Agent data
- [ ] `agent_synth.py`: tool pool → multi-agent synthesis → irrelevance negatives → dual verify.
- [ ] Export verified `Conversation`s as JSONL; importer for xLAM/ToolACE-format data.
- Exit: a few-thousand-sample verified agent dataset on disk.

## Phase 4 — SFT
- [ ] `sft.py`: loss-masked chat+tool training, function masking.
- [ ] Tool format wired through tokenizer + parser.
- Exit: model emits valid `<tool_call>`s and abstains on irrelevant queries.

## Phase 5 — Eval harness
- [ ] `tool_eval.py` AST scoring (single/parallel/multiple/irrelevance) + `text_eval.py`.
- [ ] Multi-turn simulated-user + tool-sandbox harness; JSON reports.
- Exit: a reproducible scorecard for any checkpoint.

## Phase 6 — Distillation
- [ ] `distill.py`: off-policy seq-KD (default) + on-policy reverse-KL (MiniLLM/OPD); teacher backend.
- [ ] Trajectory-level distillation with first-thought prefix.
- Exit: distilled student beats SFT-only baseline on the tool-eval scorecard.

## Phase 7 — Agent runtime + memory + demos
- [ ] `agent/runtime.py` loop; `agent/memory.py` two-tier memory as tools.
- [ ] `chat_cli.py` and `web/app.py` visualizing the tool/memory trace.
- Exit: hold a multi-turn, tool-using, memory-backed conversation locally.

## Phase 8 — Data flywheel
- [ ] Conversation store (SQLite) + feedback schema (AITL).
- [ ] `flywheel.py`: ingest → mine → verify → append → schedule retrain → eval → redeploy.
- Exit: a logged session produces verified new training samples and a retrain run.

## Phase 9 — Export (CPU/GPU/NPU)
- [ ] `export/`: GGUF, ONNX, ExecuTorch converters + Q4 quantizer + parity tests.
- Exit: same weights run via PyTorch, llama.cpp, ONNX Runtime, and ExecuTorch with matching outputs.

## Phase 10 — RL (optional)
- [ ] `rl.py`: GRPO with tool-correctness + task-success reward.
- Exit: measurable lift on multi-turn task success.

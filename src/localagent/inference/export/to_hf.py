"""Export a trained LocalAgent checkpoint to a Hugging Face Hub model repo (Phase 9).

Builds a self-contained bundle — `config.json` (the ModelConfig), weights (`model.safetensors`,
falling back to `pytorch_model.bin`), the agent heads (`agent_heads.bin`: tool + pointer head),
and a `README.md` model card — then optionally pushes it with `huggingface_hub`.

The model is pure-PyTorch + byte-level, so the card documents loading via this repo's
`LocalAgentLM`/`ModelConfig` (no `transformers` dependency).

Push requires auth: an HF token via `--token`, the `HF_TOKEN` env var, or a cached `hf auth login`.
Without a token it only writes the local bundle and prints how to push.
"""

from __future__ import annotations

import json
import os

import torch

from localagent.model import LocalAgentLM, ModelConfig

_CARD = """---
license: mit
library_name: pytorch
tags: [tool-calling, agent, tiny-llm, byte-level, on-device, from-scratch]
pipeline_tag: text-generation
---

# {name} — LocalAgent ({params:.2f}M params)

A **from-scratch, byte-level** tool-calling agent model from
[LocalAgent](https://github.com/sangbumchoi/localagent). Pure PyTorch, **{params:.2f}M params**,
trained on CPU. It pairs a tiny decoder (GQA + RoPE + SwiGLU{recur}) with a **dual head**
(tool-selection classifier + pointer/copy argument head) and **prompt-grounded constrained
decoding** for reliable tool calls across {ntools} tools (general assistant, the Claude Code /
Codex coding surface, and computer-use / productivity tools), including parallel two-call turns.

## Architecture
- vocab {vocab} (byte-level), d_model {d_model}, layers {n_layers}{loops}, heads {n_heads}/{n_kv_heads} (GQA), ffn {ffn}
- factorized embeddings: {factorized}

## Files
- `config.json` — `ModelConfig`
- `model.safetensors` / `pytorch_model.bin` — decoder weights
- `agent_heads.bin` — trained tool-selection + pointer heads (optional)

## What it can do (use cases)
One byte-level model that turns a natural-language turn into a grounded tool call — across an
assistant, a coding agent, computer-use/productivity apps, and **parallel two-call** turns:

| you say | it calls |
|---|---|
| "What's the weather in Cusco?" | `get_weather(city="Cusco")` |
| "What is 19 * 19 * 5?" | `calculator(expression="19*19*5")` |
| "Open the file bin/run.sh." | `read_file(path="bin/run.sh")` |
| "Grep for 'TODO'." | `grep_search(pattern="TODO")` |
| "Run the tests." | `run_tests()` |
| "Commit with message 'fix bug'." | `git_commit(message="fix bug")` |
| "Send an email to Greta." | `send_email(recipient="Greta")` |
| "Go to figma.com." | `open_url(url="figma.com")` |
| "Send a Slack message saying 'ship it'." | `slack_send(message="ship it")` |
| "Create a Jira ticket titled 'broken link'." | `jira_issue(summary="broken link")` |
| "Compose an email to Judy **and** search for how tall is Everest." | `send_email(recipient="Judy")` + `web_search(query="how tall is Everest")` |

Multi-turn coding (grounds a follow-up arg from a tool response):
`read_file(tests/test_api.py)` → result → `run_tests()` → "FAILED…" → fix.
At catalog scale (100s–1000s of tools) selection is done by **retrieval** (top-k) instead of a
fixed head. See the [LocalAgent repo](https://github.com/sangbumchoi/localagent).

## Load (pure PyTorch, no transformers)
```python
import json, torch
from huggingface_hub import hf_hub_download
from localagent.model import LocalAgentLM, ModelConfig

cfg_d = json.load(open(hf_hub_download("{repo}", "config.json")))
cfg = ModelConfig(**{{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__}})
model = LocalAgentLM(cfg)
from safetensors.torch import load_file
model.load_state_dict(load_file(hf_hub_download("{repo}", "model.safetensors")))
model.eval()
```
See the LocalAgent repo for the grounded decoder / agent runtime (tool head, pointer head,
retrieval, parallel-call decode).
"""


def export_hf(checkpoint: str, out_dir: str, repo_id: str | None = None, token: str | None = None,
              private: bool = True, push: bool = False) -> str:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    os.makedirs(out_dir, exist_ok=True)

    json.dump({"model_type": "localagent",
               "architecture": "LocalAgentLM (byte-level GQA+RoPE+SwiGLU)", **cfg_d},
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    sd = ck["state_dict"]
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()},
                  os.path.join(out_dir, "model.safetensors"))
    except Exception:
        torch.save(sd, os.path.join(out_dir, "pytorch_model.bin"))

    heads = {k: ck[k] for k in ("tool_head", "ptr_head") if ck.get(k)}
    if heads:
        torch.save(heads, os.path.join(out_dir, "agent_heads.bin"))

    n = sum(v.numel() for k, v in sd.items() if "lm_head" not in k)  # tied-aware-ish estimate
    card = _CARD.format(
        name=cfg.name, params=cfg.estimate_params() / 1e6, vocab=cfg.vocab_size,
        d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
        ffn=cfg.ffn_hidden, factorized=cfg.factorized,
        recur=(" + depth-recurrence" if cfg.n_loops > 1 else ""),
        loops=(f" x{cfg.n_loops} loops" if cfg.n_loops > 1 else ""),
        ntools=21, repo=repo_id or "<your-repo>")
    open(os.path.join(out_dir, "README.md"), "w").write(card)

    if not push:
        return out_dir
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("push=True but no token (pass --token or set HF_TOKEN / hf auth login)")
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")
    HfApi().upload_folder(folder_path=out_dir, repo_id=repo_id, token=token, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def export(checkpoint: str, out_path: str) -> None:  # uniform with other export/* modules
    print(export_hf(checkpoint, out_path, push=False))

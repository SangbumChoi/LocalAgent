"""KV-cached sampling (Phase 1/7): greedy / temperature / top-p.

Used by the agent runtime and eval. The forward pass lives in model/transformer.py; this adds
incremental decoding with a KV cache and stop-token handling (`<|eot|>`).
"""

from __future__ import annotations


def generate(model, tokenizer, prompt_ids, max_new_tokens: int = 256, temperature: float = 0.8,
             top_p: float = 0.95, stop_token: str = "<|eot|>"):
    raise NotImplementedError("TODO(phase-1): KV-cached incremental decode + top-p + stop token")

"""Tokenizer with agent special tokens (Phase 1).

Two modes, selected by the model tier (see docs/ARCHITECTURE_IDEAS.md):
  * ``byte``  — tokenizer-free, vocab 256 (+ specials). Used by the ultra-tiny (~1M) tier so the
    model pays no embedding tax. Encoding is just UTF-8 bytes.
  * ``bpe``   — byte-level BPE trained with the `tokenizers` lib (vocab ~32k). Used by tiny/small.

Either way the special tokens below make tool use in-vocabulary so the model can natively
emit/parse tool calls.
"""

from __future__ import annotations

SPECIAL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|eot|>",
    "<|pad|>",
]


class Tokenizer:
    """Thin wrapper around a trained byte-level BPE.

    TODO(phase-1): implement train()/encode()/decode() over the `tokenizers` lib and
    persist to a single tokenizer.json. Keep it dependency-light.
    """

    def __init__(self, path: str | None = None):
        self._tk = None  # tokenizers.Tokenizer once loaded
        self.path = path
        if path is not None:
            self.load(path)

    @classmethod
    def train(cls, corpus_files: list[str], vocab_size: int, out_path: str) -> "Tokenizer":
        raise NotImplementedError("TODO(phase-1): train byte-level BPE + add SPECIAL_TOKENS")

    def load(self, path: str) -> None:
        raise NotImplementedError("TODO(phase-1): load tokenizer.json")

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("TODO(phase-1)")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("TODO(phase-1)")

    def token_id(self, token: str) -> int:
        raise NotImplementedError("TODO(phase-1): lookup special-token id, e.g. '<|eot|>'")

"""Tokenizer — the text<->vector index map (Phase 1, implemented).

The ultra-tiny tier is **byte-level**: vocab is exactly 256 (one id per UTF-8 byte), so the
model pays no embedding tax and needs no trained tokenizer. Agent markers (``<|user|>``,
``<tool_call>`` …) are just literal UTF-8 text the model learns to emit — they stay in the 256
byte space. Byte ``0x00`` is reserved as EOS/PAD (it never appears in valid UTF-8 of our data).

A BPE mode (vocab ~32k) for the tiny/small tiers is stubbed at the bottom (Phase 1 follow-up).
"""

from __future__ import annotations

EOS_ID = 0      # reserved byte, end-of-sequence
PAD_ID = 0      # same byte doubles as pad (masked out of the loss)

# Literal text markers used to frame conversations (plain bytes, not new vocab ids).
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL = "<|tool|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

SPECIAL_MARKERS = [
    USER, ASSISTANT, TOOL,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE,
]


class ByteTokenizer:
    """UTF-8 byte tokenizer. vocab_size == 256."""

    vocab_size = 256
    eos_id = EOS_ID
    pad_id = PAD_ID

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            if stop_at_eos and i == EOS_ID:
                break
            out.append(i)
        return bytes(out).decode("utf-8", errors="replace")


def load_tokenizer(kind: str = "byte"):
    if kind == "byte":
        return ByteTokenizer()
    raise NotImplementedError("TODO(phase-1): BPE tokenizer for the tiny/small (32k vocab) tiers")

"""Render Samples into token ids for training (Phase 3/4).

SFT target framing (byte-level, markers are literal text):
    <|user|>{prompt}<|assistant|>{body}<EOS>
where body is `<tool_call>{json}</tool_call>` for tool samples or the plain text for text
samples. The loss is masked over the prompt (we only learn the assistant body + EOS).
"""

from __future__ import annotations

from localagent.data.agent_synth import Sample
from localagent.model.tokenizer import (
    ASSISTANT,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    USER,
)

IGNORE = -100


def assistant_body(s: Sample) -> str:
    if s.kind == "tool":
        return f"{TOOL_CALL_OPEN}{s.target}{TOOL_CALL_CLOSE}"
    return s.target


def prompt_text(s: Sample) -> str:
    return f"{USER}{s.prompt}{ASSISTANT}"


def render_sft(s: Sample, tok) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) of equal length; labels masked over the prompt."""
    p = tok.encode(prompt_text(s))
    b = tok.encode(assistant_body(s)) + [tok.eos_id]
    ids = p + b
    labels = [IGNORE] * len(p) + b
    return ids, labels


def render_full_text(s: Sample) -> str:
    """Full conversation text (for pretraining as a plain LM stream)."""
    return prompt_text(s) + assistant_body(s)


def build_pretrain_stream(samples: list[Sample], tok) -> list[int]:
    stream: list[int] = []
    for s in samples:
        stream.extend(tok.encode(render_full_text(s)))
        stream.append(tok.eos_id)  # document separator
    return stream

"""Render Samples into token ids for training (Phase 3/4).

SFT target framing (byte-level, markers are literal text):
    <|user|>{prompt}<|assistant|>{body}<EOS>
where body is `<tool_call>{json}</tool_call>` for tool samples or the plain text for text
samples. The loss is masked over the prompt (we only learn the assistant body + EOS).
"""

from __future__ import annotations

import json

from localagent.data.agent_synth import Sample
from localagent.data.schema import Conversation, Role
from localagent.model.tokenizer import (
    ASSISTANT,
    TOOL,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    USER,
)

IGNORE = -100


def _canon(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, separators=(",", ":"), sort_keys=True)


def render_conversation(conv: Conversation, tok) -> tuple[list[int], list[int]]:
    """Render a multi-turn Conversation to (input_ids, labels); loss is on every assistant turn
    (tool calls + final text + per-turn EOS). User and tool-response tokens are masked."""
    ids: list[int] = []
    labels: list[int] = []

    def add(text: str, learn: bool):
        t = tok.encode(text)
        ids.extend(t)
        labels.extend(t if learn else [IGNORE] * len(t))

    for m in conv.messages:
        if m.role == Role.user:
            add(USER + m.content, False)
        elif m.role == Role.tool:
            add(TOOL + TOOL_RESPONSE_OPEN + (m.tool_response or "") + TOOL_RESPONSE_CLOSE, False)
        elif m.role == Role.assistant:
            add(ASSISTANT, False)  # marker is part of the prompt
            if m.tool_calls:
                c = m.tool_calls[0]
                body = TOOL_CALL_OPEN + _canon(c.name, c.arguments) + TOOL_CALL_CLOSE
            else:
                body = m.content
            b = tok.encode(body) + [tok.eos_id]
            ids.extend(b)
            labels.extend(b)  # learn the assistant body + end-of-turn
    return ids, labels


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

"""Prompt-grounded constrained decoding (ARCHITECTURE_IDEAS §2a/2b).

A <100M byte model reliably learns tool-call *structure* but not generalizable *slot copying*.
The robust production answer (used for small-model function calling) is to not free-generate the
arguments: propose a small set of candidate calls whose arguments are **grounded in spans of the
prompt** + the tool schema, then let the model *rank* them by likelihood. Structure is guaranteed
valid; the model only has to choose. Scoring is teacher-forced (one forward per candidate — no
autoregression), so it is fast and deterministic.

The span extractors below are intentionally simple/heuristic (template-aware). A general version
would enumerate prompt n-grams per typed slot; that is the next step.
"""

from __future__ import annotations

import json
import re

import torch
import torch.nn.functional as F

from localagent.data.schema import ToolSpec
from localagent.model.tokenizer import TOOL_CALL_CLOSE, TOOL_CALL_OPEN


def _canon(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, separators=(",", ":"), sort_keys=True)


def _caps(prompt: str) -> list[str]:
    """Capitalized words excluding the first token (slot fillers: cities, names)."""
    toks = prompt.split()
    return [re.sub(r"[^A-Za-z]", "", w) for w in toks[1:] if re.match(r"[A-Z][a-z]", w)]


def _expr(prompt: str) -> str | None:
    m = re.search(r"\d+\s*[-+*]\s*\d+(?:\s*[-+*]\s*\d+)?", prompt)
    return re.sub(r"\s+", "", m.group(0)) if m else None


def _tail(prompt: str, triggers: list[str]) -> list[str]:
    """Span after a trigger word, trailing filler/punctuation stripped (queries, goals)."""
    out = []
    low = prompt.lower()
    for tg in triggers:
        i = low.find(tg)
        if i >= 0:
            span = prompt[i + len(tg):].strip()
            span = re.sub(r"\s*(online|please)?\s*[.?!]*$", "", span, flags=re.I).strip()
            if span:
                out.append(span)
    return out


def candidates(prompt: str, tools: list[ToolSpec]) -> list[tuple[str, bool, str]]:
    """Return (assistant_text, is_tool, group) candidates grounded in the prompt."""
    names = {t.name for t in tools}
    caps = _caps(prompt)
    out: list[tuple[str, bool, str]] = []
    low = prompt.lower()

    # Clear text/abstention intent -> text-only candidates (avoids e.g. "hello to X" firing
    # the planner's " to " trigger). Intent from the cue, slot (name) copied from the prompt.
    if "hello" in low:
        return [(f"Hello, {nm}!", False, "text") for nm in caps] or [("Hello!", False, "text")]
    if "morning" in low or "greet" in low:
        return [(f"Good morning, {nm}!", False, "text") for nm in caps] or [("Good morning!", False, "text")]
    if "your name" in low:
        return [("I am LocalAgent.", False, "text")]
    if "thank" in low:
        return [("You're welcome!", False, "text")]

    def tool(name, args, group):
        body = f"{TOOL_CALL_OPEN}{_canon(name, args)}{TOOL_CALL_CLOSE}"
        out.append((body, True, group))

    if "get_weather" in names:
        unit_opts = [None]
        if re.search(r"celsius", prompt, re.I):
            unit_opts.append("c")
        if re.search(r"fahrenheit", prompt, re.I):
            unit_opts.append("f")
        for city in caps:
            for u in unit_opts:
                args = {"city": city} if u is None else {"city": city, "unit": u}
                tool("get_weather", args, "tool_call")
    if "calculator" in names and (e := _expr(prompt)):
        tool("calculator", {"expression": e}, "tool_call")
    if "web_search" in names:
        for q in _tail(prompt, ["for ", "about ", "up "]):
            tool("web_search", {"query": q}, "web_search")
    if "planner" in names:
        for g in _tail(prompt, [" me ", " to "]):
            tool("planner", {"goal": g}, "planner")

    if not out:  # nothing tool-like matched -> safe text fallback
        out.append(("I am LocalAgent.", False, "text"))
    return out


@torch.no_grad()
def _score(model, tok, prompt: str, body: str, device) -> float:
    """Mean log-prob of `body` (+EOS) given `prompt`, teacher-forced."""
    pid = tok.encode(prompt)
    bid = tok.encode(body) + [tok.eos_id]
    full = torch.tensor([pid + bid], dtype=torch.long, device=device)
    logits, _ = model(full[:, :-1])
    logp = F.log_softmax(logits[0], dim=-1)
    tgt = full[0, 1:]
    lp = logp[torch.arange(tgt.shape[0]), tgt]
    return lp[len(pid) - 1:].mean().item()  # mean over body tokens (length-normalized)


def grounded_decode(model, tok, prompt: str, tools: list[ToolSpec], device="cpu") -> str:
    """Return the highest-scoring grounded assistant text (with <tool_call> tags if a tool)."""
    cands = candidates(prompt, tools)
    if not cands:
        return ""
    best = max(cands, key=lambda c: _score(model, tok, prompt, c[0], device))
    return best[0]

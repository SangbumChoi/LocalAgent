"""Prompt-grounded constrained decoding (ARCHITECTURE_IDEAS §2a/2b) — schema-driven.

A <100M byte model reliably learns tool-call *structure* + *tool selection* but not generalizable
slot *copying*. So we don't free-generate the arguments: we ground them in the prompt and let the
model *rank*. This version is **schema-driven and trigger-free** (no hardcoded per-tool phrases):

  1. tool selection  — the model free-generates a call; we parse the tool *name* (it gets that
     right even when the argument bytes are garbled).
  2. argument grounding — for the selected tool's schema args, propose candidate values generically
     from the prompt: word n-grams for strings, regex for numbers/arithmetic, enum members for
     enums, plus "omit" for optional args. Build every valid candidate call.
  3. ranking — score each candidate (teacher-forced, length-normalized) and take the best.

Open-ended *text* responses are not extractive; a light intent classifier handles them here as a
placeholder for the dual text-head proposed in ARCHITECTURE_IDEAS §2a.
"""

from __future__ import annotations

import itertools
import json
import re

import torch
import torch.nn.functional as F

from localagent.data.schema import ToolSpec
from localagent.model.tokenizer import TOOL_CALL_CLOSE, TOOL_CALL_OPEN

MAX_SPAN_WORDS = 8
MAX_COMBOS = 240


def _canon(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, separators=(",", ":"), sort_keys=True)


def _spans(prompt: str) -> list[str]:
    """All contiguous word n-grams (1..MAX_SPAN_WORDS), edge punctuation stripped. Generic — no
    knowledge of which words are slots."""
    words = prompt.split()
    out, seen = [], set()
    for i in range(len(words)):
        for j in range(i + 1, min(i + MAX_SPAN_WORDS, len(words)) + 1):
            s = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9!]+$", "", " ".join(words[i:j]))
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _arith(prompt: str) -> list[str]:
    m = re.search(r"\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*", prompt)
    return [re.sub(r"\s+", "", m.group(0))] if m else []


def _numbers(prompt: str) -> list[str]:
    return re.findall(r"-?\d+", prompt)


def _arg_options(prompt: str, name: str, schema: dict, required: bool) -> list:
    """Candidate values for one argument, from its JSON-schema type (generic)."""
    if "enum" in schema:
        opts = list(schema["enum"])
    elif schema.get("format") == "arithmetic" or "express" in name:
        opts = _arith(prompt)
    elif schema.get("type") in ("integer", "number"):
        opts = _numbers(prompt)
    else:  # string / unknown -> prompt spans
        opts = _spans(prompt)
    if not required:
        opts = [None] + opts          # allow omitting optional args
    return opts or ([None] if not required else [])


def _tool_bodies(prompt: str, tool: ToolSpec) -> list[str]:
    props = (tool.parameters or {}).get("properties", {})
    required = set((tool.parameters or {}).get("required", []))
    names = list(props.keys())
    per_arg = [_arg_options(prompt, n, props[n], n in required) for n in names]
    bodies = []
    for combo in itertools.islice(itertools.product(*per_arg), MAX_COMBOS):
        args = {n: v for n, v in zip(names, combo) if v is not None}
        if all(n in args for n in required):
            bodies.append(f"{TOOL_CALL_OPEN}{_canon(tool.name, args)}{TOOL_CALL_CLOSE}")
    return bodies


# ---- open-text intent (placeholder for the dual text-head, ARCHITECTURE_IDEAS §2a) ----
def _text_candidates(prompt: str) -> list[str] | None:
    low = prompt.lower()
    caps = [re.sub(r"[^A-Za-z]", "", w) for w in prompt.split()[1:] if re.match(r"[A-Z][a-z]", w)]
    if "hello" in low:
        return [f"Hello, {nm}!" for nm in caps] or ["Hello!"]
    if "morning" in low or "greet" in low:
        return [f"Good morning, {nm}!" for nm in caps] or ["Good morning!"]
    if "your name" in low:
        return ["I am LocalAgent."]
    if "thank" in low:
        return ["You're welcome!"]
    return None


@torch.no_grad()
def _best(model, tok, prompt: str, bodies: list[str], device) -> str:
    """Length-normalized log-prob of each body given the prompt, scored in ONE batched forward
    (candidates share the prompt prefix), returning the argmax body."""
    pid = tok.encode(prompt)
    seqs = [pid + tok.encode(b) + [tok.eos_id] for b in bodies]
    maxlen = max(len(s) for s in seqs)
    X = torch.full((len(seqs), maxlen), tok.pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        X[i, : len(s)] = torch.tensor(s, device=device)
    logits, _ = model(X[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    best_i, best_s = 0, -1e9
    for i, s in enumerate(seqs):
        sc = tok_lp[i, len(pid) - 1: len(s) - 1].mean().item()
        if sc > best_s:
            best_i, best_s = i, sc
    return bodies[best_i]


def _preselect_tool(model, tok, prompt: str, names: set[str], device) -> str | None:
    """Free-generate a call and read the tool name (the model gets selection right)."""
    from localagent.agent.parser import extract_tool_calls
    from localagent.model.tokenizer import ASSISTANT, USER
    from localagent.inference.generate import generate
    framed = f"{USER}{prompt}{ASSISTANT}"
    gen, _ = generate(model, tok, framed, max_new_tokens=80, temperature=0.0)
    calls = extract_tool_calls(gen)
    return calls[0].name if calls and calls[0].name in names else None


def candidates(prompt: str, tools: list[ToolSpec]) -> list[tuple[str, bool, str]]:
    """All grounded candidates as (text, is_tool, group). Used by tests; ranking picks one."""
    group_of = {"get_weather": "tool_call", "calculator": "tool_call",
                "web_search": "web_search", "planner": "planner"}
    txt = _text_candidates(prompt)
    if txt is not None:
        return [(t, False, "text") for t in txt]
    out = []
    for tool in tools:
        for b in _tool_bodies(prompt, tool):
            out.append((b, True, group_of.get(tool.name, "tool_call")))
    return out or [("I am LocalAgent.", False, "text")]


def grounded_decode(model, tok, prompt: str, tools: list[ToolSpec], device="cpu") -> str:
    # 1. open-text intent first (placeholder text-head)
    txt = _text_candidates(prompt)
    if txt is not None:
        return _best(model, tok, prompt, txt, device)
    # 2. schema-driven grounded tool candidates, narrowed by the model's tool pick
    names = {t.name for t in tools}
    picked = _preselect_tool(model, tok, prompt, names, device)
    use = [t for t in tools if t.name == picked] if picked else tools
    bodies = []
    for t in use:
        bodies += _tool_bodies(prompt, t)
    if not bodies:
        return "I am LocalAgent."
    return _best(model, tok, prompt, bodies, device)

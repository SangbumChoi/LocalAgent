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

MAX_COMBOS = 60
# Generic English prepositions that introduce a slot value ("weather IN Paris", "search FOR X",
# "plan TO learn guitar"). Not tool-specific — works across schemas. ("me" excluded: it nests
# badly, e.g. "remind ME to call" vs "to call".)
PREPS = ["for", "about", "up", "to", "in", "on", "of"]


def _canon(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, separators=(",", ":"), sort_keys=True)


def _strip(s: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|\s*(online|please|right now)?\s*[.?!]*$", "", s, flags=re.I).strip()


# Argument names whose value is a proper-noun entity (take the capitalized span) vs free text
# (take the whole tail, which may itself contain a proper noun, e.g. query "capital of Peru").
ENTITY_ARGS = {"city", "location", "name", "person", "artist", "song", "album", "place",
               "recipient"}


def _best_string(prompt: str, arg: str = "") -> str:
    """Deterministic string-slot value, arg-aware (the tool head already chose the tool/arg):
    entity args -> first capitalized proper-noun span; free-text args -> longest preposition tail
    (else the imperative tail after the leading verb). Generic English heuristics, not per-tool."""
    caps = re.findall(r"(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*", " ".join(prompt.split()[1:]))
    low = prompt.lower()
    tails = [_strip(prompt[i + len(p) + 2:]) for p in PREPS if (i := low.find(f" {p} ")) >= 0]
    tails = [t for t in tails if t]
    if arg in ENTITY_ARGS and caps:
        return _strip(caps[0])
    if tails:
        return max(tails, key=len)
    if caps:
        return _strip(caps[0])
    words = prompt.split()  # imperative "Define X." / "Play X." -> drop the leading verb
    return _strip(" ".join(words[1:])) if len(words) > 1 else _strip(prompt)


def _arith(prompt: str) -> list[str]:
    m = re.search(r"\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*", prompt)
    return [re.sub(r"\s+", "", m.group(0))] if m else []


def _numbers(prompt: str) -> list[str]:
    return re.findall(r"-?\d+", prompt)


def _quoted(prompt: str) -> list[str]:
    """Content of the first single/double-quoted span (patterns, commands, commit messages)."""
    m = re.search(r"'([^']+)'|\"([^\"]+)\"", prompt)
    return [next(g for g in m.groups() if g)] if m else []


def _path(prompt: str) -> list[str]:
    """First file-path/-name token (has a slash or a file extension)."""
    m = re.search(r"[A-Za-z0-9_.\-/]+/[A-Za-z0-9_.\-/]*|[A-Za-z0-9_.\-/]+\.[A-Za-z0-9]{1,5}\b",
                  prompt)
    return [m.group(0).rstrip(".")] if m else []


def _url(prompt: str) -> list[str]:
    """First URL/domain token (optionally with scheme/path)."""
    m = re.search(r"(?:https?://)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[\w./-]*)?", prompt)
    return [m.group(0).rstrip(".")] if m else []  # drop trailing sentence period


def _arg_options(prompt: str, name: str, schema: dict, required: bool, ptr=None) -> list:
    """Candidate values for one argument. If a pointer head is given (`ptr`), string-typed args
    are filled by its learned copy span; otherwise schema/heuristic extractors are used."""
    from localagent.agent.pointer_head import ARG_IDX
    fmt = schema.get("format")
    if "enum" in schema:
        opts = list(schema["enum"])
    elif fmt == "arithmetic" or "express" in name:
        opts = _arith(prompt)
    elif ptr is not None and name in ARG_IDX:        # learned pointer/copy span
        ph, feats_row, framed_ids, tok = ptr
        s, e = ph.predict_span(feats_row, name)
        opts = [tok.decode(framed_ids[s:e + 1])]
    elif fmt == "quoted":
        opts = _quoted(prompt)
    elif fmt == "path":
        opts = _path(prompt)
    elif fmt == "url":
        opts = _url(prompt)
    elif schema.get("type") in ("integer", "number"):
        opts = _numbers(prompt)
    else:  # string / unknown -> deterministic best prompt span (arg-aware)
        opts = [_best_string(prompt, name)]
    if not required:
        opts = [None] + opts          # allow omitting optional args
    return opts or ([None] if not required else [])


def _tool_bodies(prompt: str, tool: ToolSpec, ptr=None) -> list[str]:
    props = (tool.parameters or {}).get("properties", {})
    required = set((tool.parameters or {}).get("required", []))
    names = list(props.keys())
    per_arg = [_arg_options(prompt, n, props[n], n in required, ptr) for n in names]
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


def _preselect_tool(model, tok, prompt: str, names: set[str], device) -> str | None:
    """Let the model free-generate a call and read the tool *name* (it picks the tool reliably
    even when the argument bytes are garbled). This is the schema-agnostic tool selector; the
    grounded candidates then only have to get the arguments right."""
    from localagent.agent.parser import extract_tool_calls
    from localagent.inference.generate import generate
    from localagent.model.tokenizer import ASSISTANT, USER
    gen, _ = generate(model, tok, f"{USER}{prompt}{ASSISTANT}", max_new_tokens=80, temperature=0.0)
    calls = extract_tool_calls(gen)
    return calls[0].name if calls and calls[0].name in names else None


@torch.no_grad()
def _ctx_feats(model, tok, ctx: str, device):
    ids = tok.encode(ctx)
    _, feats = model(torch.tensor([ids], device=device), return_hidden=True)
    return feats[0], ids


def grounded_decode_parallel(model, tok, prompt: str, tools: list[ToolSpec], device="cpu",
                             tool_head=None, ptr_head=None) -> str:
    """For 'do X and Y' turns: split on ' and ', ground each conjunct, concatenate the calls."""
    parts = [p.strip() for p in prompt.split(" and ") if p.strip()]
    return "".join(grounded_decode(model, tok, p, tools, device, tool_head, ptr_head)
                   for p in parts)


def grounded_decode(model, tok, prompt: str, tools: list[ToolSpec], device="cpu",
                    tool_head=None, ptr_head=None, framed=False) -> str:
    """Grounded constrained decode. `framed=False`: `prompt` is a raw user turn (framed as
    <|user|>..<|assistant|> internally) — single-turn. `framed=True`: `prompt` is the full
    multi-turn context already ending at the assistant marker (the next action is decoded over the
    whole history, so args can be grounded in earlier tool responses)."""
    from localagent.model.tokenizer import ASSISTANT, USER
    ctx = prompt if framed else f"{USER}{prompt}{ASSISTANT}"
    score = prompt if not framed else ctx  # what _score conditions on
    feats = ids = None
    # 1. tool selection
    if tool_head is not None:
        feats, ids = _ctx_feats(model, tok, ctx, device)
        from localagent.agent.tool_head import CLASSES
        picked = CLASSES[int(tool_head(feats[-1]).argmax(-1))]
        if picked == "text":
            return _best(model, tok, score, _text_candidates(prompt) or ["I am LocalAgent."], device)
    else:
        txt = _text_candidates(prompt)
        if txt is not None:
            return _best(model, tok, score, txt, device)
        picked = _preselect_tool(model, tok, prompt, {t.name for t in tools}, device)
    # 2. fill the selected tool's args — learned pointer/copy spans if a pointer head is given.
    use = [t for t in tools if t.name == picked] if picked else tools
    ptr = None
    if ptr_head is not None:
        if feats is None:
            feats, ids = _ctx_feats(model, tok, ctx, device)
        ptr = (ptr_head, feats, ids, tok)
    bodies = []
    for t in use:
        bodies += _tool_bodies(prompt, t, ptr)
    if not bodies:
        return "I am LocalAgent."
    return _best(model, tok, score, bodies, device)

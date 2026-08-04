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
from localagent.model.tokenizer import ASSISTANT, TOOL_CALL_CLOSE, TOOL_CALL_OPEN

MAX_COMBOS = 60
# Generic English prepositions that introduce a slot value ("weather IN Paris", "search FOR X",
# "plan TO learn guitar"). Not tool-specific — works across schemas. ("me" excluded: it nests
# badly, e.g. "remind ME to call" vs "to call".)
PREPS = ["for", "about", "up", "to", "in", "on", "of"]


def _canon(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, separators=(",", ":"), sort_keys=True)


def _strip(s: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|\s*(online|please|right now)?\s*[.?!]*$", "", s, flags=re.I).strip()


def _action_tail(prompt: str) -> str:
    """Return the current-step instruction, excluding earlier goal/state slots when present."""

    match = re.search(r"Next required action:\s*(.*?)(?:\s+Last tool result:|$)", prompt, re.I)
    return match.group(1).strip() if match else prompt


# Argument names whose value is a proper-noun entity (take the capitalized span) vs free text
# (take the whole tail, which may itself contain a proper noun, e.g. query "capital of Peru").
ENTITY_ARGS = {"city", "location", "name", "person", "artist", "song", "album", "place",
               "recipient"}
PHONE_ARGS = {"phone", "phone_number", "telephone", "telephone_number", "mobile"}
TEXT_ARGS = {"content", "message", "text", "body", "subject", "title", "note", "comment"}
APP_ARGS = {"app_name"}
TARGET_ARGS = {"target"}
EMAIL_ARGS = {"to", "recipient"}
ID_ARGS = {"id", "identifier", "task_id", "user_id", "notification_id", "event_id"}
_IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9-]*_[A-Za-z0-9][A-Za-z0-9_-]*|[0-9a-f]{8}-[0-9a-f-]{27,})\b"
)


def _boolean(prompt: str) -> list[bool]:
    """Infer a JSON boolean from generic enable/disable language in the prompt."""
    low = prompt.lower()
    if re.search(r"\b(?:turn|switch|set|power)\s+off\b|\b(?:disable|disabled|false|no)\b", low):
        return [False]
    if re.search(r"\b(?:turn|switch|set|power)\s+on\b|\b(?:enable|enabled|true|yes)\b", low):
        return [True]
    return []


def _phone(prompt: str) -> list[str]:
    """Return the first phone-like span without trailing message prose."""
    # Prefer explicit international numbers.  UUIDs and timestamps in tool results also contain
    # digit/hyphen runs, so an unqualified search can silently copy ``53108174`` from a UUID.
    matches = re.findall(r"\+\d[\d ()-]{6,}\d", prompt)
    if not matches:
        matches = re.findall(r"(?<![A-Za-z0-9])\d{7,}(?![A-Za-z0-9])", prompt)
    return [re.sub(r"[ ()-]", "", matches[0])] if matches else []


def _identifier(prompt: str, arg: str = "") -> list[str]:
    """Extract a structured identifier instead of copying an entire instruction sentence."""

    stem = arg[:-3] if arg.endswith("_id") else ""
    values = _IDENTIFIER.findall(prompt)
    if stem:
        scoped = [value for value in values if value.lower().startswith(f"{stem.lower()}_")]
        if scoped:
            return scoped
    return values


def _text_arg(prompt: str, arg: str = "") -> list[str]:
    """Extract a text slot from generic delimiters or field-labelled quoted values."""
    action = _action_tail(prompt)
    goal = prompt.split(" Current state JSON:", 1)[0]
    for source in (action, goal):
        match = re.search(r"(?:saying|with message|message|text|content)\s*:\s*(.+)", source, re.I)
        if match:
            return [_strip(match.group(1))]
    low = prompt.lower()
    quoted = [value for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", action)
              for value in (left or right,)]
    if not quoted:
        quoted = [value for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", goal)
                  for value in (left or right,)]
    if arg in {"to", "recipient"} or "address field" in low or "recipient" in low:
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", prompt)
        if email:
            return [email.group(0)]
    labels = {
        "subject": r"subject(?: field)?",
        "title": r"title(?:d)?",
        "body": r"body|message",
        "content": r"content|body",
    }
    label = labels.get(arg)
    if label:
        match = re.search(r"(?:" + label + r")[^'\"]*['\"]([^'\"]+)['\"]", prompt, re.I)
        if match:
            return [match.group(1)]
    if arg in {"text", "message"}:
        action_low = action.lower()
        if "subject" in action_low and quoted:
            return [quoted[0]]
        if ("body" in action_low or "message field" in action_low) and quoted:
            return [quoted[-1]]
        if quoted:
            return [quoted[-1]]
    if arg in {"title", "subject", "label"} and quoted:
        return [quoted[0]]
    return []


def _app_name(prompt: str) -> list[str]:
    """Extract the application named after a generic launch/open/start instruction."""
    action = _action_tail(prompt)
    sources = (action,) if action != prompt else (prompt,)
    for source in sources:
        match = re.search(
            r"(?:launch|open|bring\s+up|start)\s+(?:the\s+)?([A-Z][A-Za-z0-9_.-]*)",
            source,
            re.I,
        )
        if match:
            return [match.group(1)]
    return []


def _target(prompt: str) -> list[str]:
    """Extract a semantic UI target from click/select/tap wording."""
    for source in (_action_tail(prompt), prompt):
        quoted = re.search(r"(?:click|select|tap)\s+['\"]([^'\"]+)['\"]", source, re.I)
        if quoted:
            value = quoted.group(1).strip()
            return [value if value.lower().startswith("the ") else f"the {value}"]
        match = re.search(
            r"(?:click|select|tap)\s+(?:on\s+)?((?:the\s+)?[A-Za-z][A-Za-z ]*?)(?:\s+at\s+x=|\s+on\s+(?:the\s+)?(?:phone|android)|[.!?]|$)",
            source,
            re.I,
        )
        if match:
            value = _strip(match.group(1))
            return [value if value.lower().startswith("the ") else f"the {value}"]
    return []


def _best_string(prompt: str, arg: str = "") -> str:
    """Deterministic string-slot value, arg-aware (the tool head already chose the tool/arg):
    entity args -> first capitalized proper-noun span; free-text args -> longest preposition tail
    (else the imperative tail after the leading verb). Generic English heuristics, not per-tool."""
    # Role labels are protocol scaffolding, not user content.  Strip them before the generic
    # capitalized-span heuristic and drop the imperative's first word so ``USER: Send Fredrik``
    # yields ``Fredrik`` rather than the verb ``Send``.
    source = _action_tail(prompt)
    source = re.sub(r"^(?:USER|SYSTEM|TOOL_RESULT)\s*:\s*", "", source, flags=re.I)
    words = source.split()
    caps = re.findall(r"(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*", " ".join(words[1:]))
    low = source.lower()
    tails = [_strip(source[i + len(p) + 2:]) for p in PREPS if (i := low.find(f" {p} ")) >= 0]
    tails = [t for t in tails if t]
    if arg in PHONE_ARGS:
        values = _phone(prompt)
        if values:
            return values[0]
    if arg in ID_ARGS or arg.endswith("_id"):
        values = _identifier(prompt, arg)
        if values:
            return values[0]
    if arg in EMAIL_ARGS:
        values = _text_arg(prompt, arg)
        if values:
            return values[0]
    if arg in TEXT_ARGS:
        values = _text_arg(prompt, arg)
        if values:
            return values[0]
    if arg in APP_ARGS:
        values = _app_name(prompt)
        if values:
            return values[0]
    if arg in TARGET_ARGS:
        values = _target(prompt)
        if values:
            return values[0]
    if arg in ENTITY_ARGS and caps:
        return _strip(caps[0])
    if tails:
        return max(tails, key=len)
    if caps:
        return _strip(caps[0])
    return _strip(" ".join(words[1:])) if len(words) > 1 else _strip(source)


def _arith(prompt: str) -> list[str]:
    m = re.search(r"\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*", prompt)
    return [re.sub(r"\s+", "", m.group(0))] if m else []


def _numbers(prompt: str) -> list[str]:
    return re.findall(r"-?\d+", prompt)


def _number_arg(prompt: str, name: str) -> list[str]:
    """Prefer a labelled numeric argument such as ``x=120`` over unrelated state numbers."""

    for source in (_action_tail(prompt), prompt):
        match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+(?:\.\d+)?)", source, re.I)
        if match:
            return [match.group(1)]
    return _numbers(prompt)


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
    for source in (_action_tail(prompt), prompt):
        m = re.search(r"(?:https?://)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[\w./-]*)?", source)
        if m:
            return [m.group(0).rstrip(".")]  # drop trailing sentence period
    return []


def _arg_options(prompt: str, name: str, schema: dict, required: bool, ptr=None) -> list:
    """Candidate values for one argument. If a pointer head is given (`ptr`), string-typed args
    are filled by its learned copy span; otherwise schema/heuristic extractors are used."""
    fmt = schema.get("format")
    if "enum" in schema:
        opts = list(schema["enum"])
    elif fmt == "arithmetic" or "express" in name:
        opts = _arith(prompt)
    elif schema.get("type") == "boolean":
        opts = _boolean(prompt)
    elif name in PHONE_ARGS:
        opts = _phone(prompt)
    elif name in ID_ARGS or name.endswith("_id"):
        opts = _identifier(prompt, name)
    elif ptr is not None and name in ptr[0].arg_idx:        # learned pointer/copy span
        ph, feats_row, framed_ids, tok = ptr[:4]
        span_bounds = ptr[4] if len(ptr) > 4 else None
        s, e = ph.predict_span(feats_row, name, span_bounds=span_bounds)
        opts = [tok.decode(framed_ids[s:e + 1])]
    elif name in TEXT_ARGS:
        opts = _text_arg(prompt, name)
    elif name in EMAIL_ARGS:
        opts = _text_arg(prompt, name) or [_best_string(prompt, name)]
    elif name in APP_ARGS:
        opts = _app_name(prompt)
    elif name in TARGET_ARGS:
        opts = _target(prompt)
    elif fmt == "quoted":
        # Public trajectories frequently provide semantic labels without quote marks (for
        # example, ``click the Search box``).  Keep the quoted extractor first, but fall back to
        # the generic string heuristic instead of forcing a learned pointer to copy a malformed
        # span that includes the state observation.
        opts = _quoted(prompt) or [_best_string(prompt, name)]
    elif fmt == "path":
        opts = _path(prompt)
    elif fmt == "url":
        opts = _url(prompt)
    elif schema.get("type") in ("integer", "number"):
        # cast to typed numbers so the canonical body matches the int/float target (not "5").
        cast = int if schema.get("type") == "integer" else float
        opts = [cast(n) for n in _number_arg(prompt, name)]
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
    if re.search(r"\b(?:already complete|no action|nothing to do|without invoking)\b", low):
        return ["I won't invoke a tool."]
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
    max_len = getattr(getattr(model, "cfg", None), "max_seq_len", None)
    if max_len is not None:
        # A grounded value can itself be longer than the model window (for example, a tool
        # schema copied from a long page).  Such a candidate cannot be scored safely even after
        # dropping the prompt; fail closed instead of sending an overlong tensor to RoPE.
        bodies = [b for b in bodies if len(tok.encode(b)) + 1 <= max_len]
        if not bodies:
            return "I cannot complete this request."
    seqs = [pid + tok.encode(b) + [tok.eos_id] for b in bodies]
    maxlen = max(len(s) for s in seqs)
    # Keep within the model's context window: a long multi-turn history + a candidate body can
    # exceed max_seq_len. Trim from the LEFT (drop the oldest *prompt* tokens, shared by every
    # candidate) so bodies stay intact and the scoring offsets below shift consistently.
    max_len = max_len or maxlen
    cut = min(max(0, maxlen - 1 - max_len), len(pid) - 1)
    if cut:
        seqs = [s[cut:] for s in seqs]
        pid = pid[cut:]
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
    max_len = getattr(getattr(model, "cfg", None), "max_seq_len", len(ids))
    if len(ids) > max_len:
        # Keep the newest observation/tool result.  Candidate grounding still sees the complete
        # raw prompt, while the model's RoPE/context window receives a valid bounded suffix.
        ids = ids[-max_len:]
    _, feats = model(torch.tensor([ids], device=device), return_hidden=True)
    return feats[0], ids


def _grounding_span(ids: list[int], tok, grounding: str) -> tuple[int, int] | None:
    """Locate the grounding suffix inside a catalog-framed context for pointer masking."""
    grounding_ids = tok.encode(grounding)
    if not grounding_ids:
        return None
    assistant_ids = tok.encode(ASSISTANT)
    suffix_end = len(ids) - len(assistant_ids)
    suffix_start = suffix_end - len(grounding_ids)
    if suffix_start >= 0 and ids[suffix_start:suffix_end] == grounding_ids:
        return suffix_start, suffix_end - 1
    # Boundary-aware tokenizers can differ at concatenation points; use the last occurrence as a
    # conservative fallback, which still avoids the catalog prefix in normal prompts.
    for start in range(len(ids) - len(grounding_ids), -1, -1):
        if ids[start : start + len(grounding_ids)] == grounding_ids:
            return start, start + len(grounding_ids) - 1
    return None


def hybrid_decode(model, tok, prompt: str, tools: list[ToolSpec], device="cpu", *,
                  retriever=None, route_head=None, ptr_head=None, selector=None, top_m=1, k=8,
                  framed=False, blocked_candidates: set[str] | None = None,
                  selector_first: bool = False,
                  grounding_prompt: str | None = None) -> str:
    """The *generable* decode path — no fixed-N classifier. Selection narrows the catalog to a few
    candidates, then the model RANKS their grounded bodies; argument *values* are copied by
    `ptr_head` (the one sub-task a tiny model can't free-generate). An optional 5-way `route_head`
    gates text-vs-tool up front.

    Selection source:
      `selector` (a `BoundSelector`, recommended) — a *trained* two-tower scorer that ranks every
        tool by its description embedding; we keep its top-`top_m`. Generalizes to unseen tools.
      else `retriever` — zero-training char-ngram retrieval top-k (weaker; the model must then rank).
    Either way adding a tool needs zero head reshape / retraining. ``grounding_prompt`` optionally
    separates the user/history text used for schema argument candidates from the full model
    context (which may contain a serialized function catalog)."""
    from localagent.agent.retriever import ToolRetriever
    from localagent.model.tokenizer import ASSISTANT, USER
    ctx = prompt if framed else f"{USER}{prompt}{ASSISTANT}"
    score = prompt if not framed else ctx
    grounding = prompt if grounding_prompt is None else grounding_prompt
    feats = ids = None
    # 0. route gate (text vs tool) — falls back to the heuristic text detector when no head given
    if route_head is not None or selector is not None:
        feats, ids = _ctx_feats(model, tok, ctx, device)
    if route_head is not None:
        from localagent.agent.routes import ROUTES
        if ROUTES[int(route_head(feats[-1]).argmax(-1))] == "text":
            # Fail open for prompts that contain no recognized text intent.  A small route head
            # can misclassify a long state-conditioned tool prompt as ``text``; turning that into
            # an unconditional abstention makes retries impossible and hides the selector's
            # useful tool prior.  Known greeting/identity/thanks intents still take the text path.
            text_candidates = _text_candidates(grounding)
            if text_candidates is not None:
                return _best(model, tok, score, text_candidates, device)
    else:
        txt = _text_candidates(grounding)
        if txt is not None:
            return _best(model, tok, score, txt, device)
    # 1. selection: trained dense selector (top-m) if given, else retrieval top-k
    selector_order: list[str] | None = None
    if selector is not None:
        selector_order = selector.rank(feats[-1], allowed_names={t.name for t in tools})
        keep = set(selector_order[:top_m])
    else:
        retriever = retriever or ToolRetriever(tools)
        keep = set(retriever.retrieve(grounding, k=k))
    use = [t for t in tools if t.name in keep] or tools
    if selector_order is not None:
        order = {name: index for index, name in enumerate(selector_order)}
        use.sort(key=lambda tool: order.get(tool.name, len(order)))
    # 2. argument values via learned pointer/copy spans
    ptr = None
    if ptr_head is not None:
        if feats is None:
            feats, ids = _ctx_feats(model, tok, ctx, device)
        bounds = _grounding_span(ids, tok, grounding) if grounding_prompt is not None else None
        ptr = (ptr_head, feats, ids, tok, bounds) if bounds is not None else (ptr_head, feats, ids, tok)
    # 3. rank every candidate's grounded body; _best picks the tool AND args jointly
    bodies = []
    for t in use:
        bodies += _tool_bodies(grounding, t, ptr)
    if not bodies:
        return "I am LocalAgent."
    if blocked_candidates:
        available = [body for body in bodies if body not in blocked_candidates]
        # If every grounded candidate was rejected, fail open and let the model retry rather than
        # returning an unrelated abstention.  The runtime's bounded attempt budget still limits
        # repeated calls, while the common case gets a genuine alternative candidate.
        bodies = available or bodies
    if selector_first:
        return bodies[0]
    return _best(model, tok, score, bodies, device)


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

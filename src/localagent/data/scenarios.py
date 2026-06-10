"""SOTA-agent SCENARIO corpus (Codex / Claude Code behaviour families).

Why this module exists: ``paraphrase.py`` and ``contextual.py`` teach *which tool* a single
request maps to. But modern coding/research agents also have to decide *when NOT to act yet*
(ask a clarifying question), *when not to act at all* (over-trigger negatives), *when to fan out*
(parallel calls), and *how to carry data across turns* (chained / error-recovery trajectories).
This source adds exactly those four single-turn behaviours and three multi-turn families on top
of the existing generators, reusing their primitives verbatim.

Contracts honored (other code depends on these):
  * Reuses the existing ``Sample`` dataclass and ``_tool_target`` canonical-JSON helper, and the
    schema ``Conversation``/``Message``/``Role``/``ToolCall`` — schema/configs/render untouched.
  * Parallel samples carry the SAME shape ``agent_synth.parallel`` produces: ``calls`` is a list
    of ``{"name","arguments"}`` dicts (len >= 2), ``target`` is the space-joined ``_tool_target``
    of each call, and ``ref_name``/``ref_args`` are the FIRST call.
  * Pointer-copy grounding: every tool-call arg value that is COPIED appears as a literal
    substring of the relevant prior turn — the USER prompt for a first call, and the prior
    ``tool_response`` for chained / error-recovery follow-ups.
  * All tools referenced are in ``localagent.agent.toolset.STANDARD_TOOLS``.
  * Train vs eval are DISJOINT in BOTH phrasing skeletons AND slot values.

Public API:
  * ``scenario_samples(n=1, seed=0, split="train") -> list[Sample]`` — single-turn additions,
    balanced across {clarify, abstain, parallel}; ``n`` is the count PER sub-type.
  * ``scenario_episodes(n=1, seed=0, split="train") -> list[Conversation]`` — multi-turn
    episodes, balanced across {workflow, chained, error_recovery}; ``n`` is the count PER family.
"""

from __future__ import annotations

import json
import random

from localagent.data.agent_synth import Sample, _tool_target
from localagent.data.schema import Conversation, Message, Role, ToolCall

# ---------------------------------------------------------------------------------------------
# Slot pools — DISJOINT train/eval (module-local, so the eval pool here never overlaps train).
# Values ground as literal substrings under any skeleton; quoted-format args carry their quotes
# inside the phrasing, paths/urls appear verbatim.
# ---------------------------------------------------------------------------------------------

# code paths (read_file / edit_file / grep targets). Disjoint train/eval.
S_PATHS = {
    "train": ["app/core.py", "lib/parse.py", "src/server.ts", "pkg/store.go", "web/view.jsx",
              "utils/net.py", "api/users.py", "services/cart.py", "db/migrate.sql", "cli/main.rs"],
    "eval": ["mod/auth.py", "ui/panel.tsx", "svc/billing.py", "net/socket.go", "data/etl.py"],
}
# grep patterns / symbols. Disjoint train/eval.
S_PATTERNS = {
    "train": ["compute_total", "RetryQueue", "parse_token", "MAX_RETRIES", "load_config",
              "render_row", "validate_input", "build_index", "flush_buffer", "open_session"],
    "eval": ["decode_frame", "RateGuard", "merge_shards", "MIN_BATCH", "warm_cache"],
}
# shell commands (run_command). Disjoint train/eval.
S_COMMANDS = {
    "train": ["make build", "npm run compile", "cargo check", "go vet ./...", "make migrate",
              "yarn lint", "tox -e py311"],
    "eval": ["make package", "npm run typecheck", "cargo clippy"],
}
# commit messages (git_commit). Disjoint train/eval.
S_COMMITS = {
    "train": ["fix null deref", "tighten validation", "cache the lookup", "guard empty input",
              "split the helper", "log the retry"],
    "eval": ["debounce the click", "trim trailing space", "pin the toolchain"],
}
# research queries (web_search). Disjoint train/eval.
S_QUERIES = {
    "train": ["rust async runtimes", "postgres index types", "kafka consumer groups",
              "wasm component model", "grpc streaming patterns", "sqlite wal mode"],
    "eval": ["zig comptime basics", "duckdb vs sqlite", "redis stream consumers"],
}
# result URLs returned by a search (open_url follow-up). Disjoint train/eval.
S_URLS = {
    "train": ["docs.rs/tokio", "wiki.postgresql.org/Indexes", "kafka.apache.org/intro",
              "webassembly.org/specs", "grpc.io/docs/guides"],
    "eval": ["ziglang.org/learn", "duckdb.org/docs", "redis.io/docs/streams"],
}
# cities for parallel weather batches. Disjoint train/eval.
S_CITIES = {
    "train": ["Turin", "Bergen", "Cork", "Aarhus", "Ghent", "Bilbao", "Graz", "Nantes"],
    "eval": ["Bratislava", "Cluj", "Leuven", "Aalborg"],
}
# directory names for parallel make_dir batches. Disjoint train/eval.
S_DIRS = {
    "train": ["build/cache", "tmp/work", "out/dist", "logs/jobs", "data/raw", "var/state"],
    "eval": ["dist/web", "scratch/run", "cache/blobs"],
}

# Clarify: underspecified request -> short follow-up QUESTION (no tool, kind="text").
# Skeletons disjoint train/eval; the target question is the canonical clarification.
CLARIFY = {
    "train": [
        ("Email the team.", "Who should I email, and what should the message say?"),
        ("Book a meeting.", "What time, and with whom?"),
        ("Schedule a call.", "When should the call be, and who is on it?"),
        ("Create a reminder.", "What should I remind you about, and when?"),
        ("Open the file.", "Which file should I open?"),
        ("Run the script.", "Which script should I run?"),
        ("Make a reservation.", "Where, for how many people, and at what time?"),
        ("Add it to my calendar.", "What is the event, and when is it?"),
    ],
    "eval": [
        ("Send the report.", "Which report, and who should receive it?"),
        ("Deploy it.", "Which service should I deploy, and to which environment?"),
        ("Translate this.", "What text should I translate, and into which language?"),
        ("Rename the file.", "Which file, and what should the new name be?"),
        ("Cancel my subscription.", "Which subscription would you like me to cancel?"),
    ],
}

# Abstain: NO tool should fire (chit-chat, direct knowledge, tool-ish-looking statements that
# are reports of already-done work). target is the plain assistant reply (no tool, kind="text").
ABSTAIN = {
    "train": [
        ("Thanks, that's really helpful!", "Glad I could help!"),
        ("I already ran the tests, thanks.", "Great, sounds like you're all set."),
        ("Nice, the deploy worked.", "Awesome, glad the deploy went smoothly!"),
        ("How are you doing today?", "I'm doing well, thanks for asking!"),
        ("What can you help me with?", "I can help with coding, research, and everyday tasks."),
        ("I just committed the fix myself.", "Nice work getting that committed."),
        ("Good morning!", "Good morning! How can I help?"),
        ("That makes sense, thank you.", "You're welcome!"),
        ("I'm going to grab a coffee.", "Enjoy your coffee!"),
        ("We already merged the PR.", "Great, glad the PR is merged."),
    ],
    "eval": [
        ("Have a great weekend!", "Thanks, you too!"),
        ("I finished writing the docs already.", "Nice, glad the docs are done."),
        ("Cool, the build passed on its own.", "Great to hear the build is green."),
        ("Just wanted to say hi.", "Hi there! How can I help?"),
        ("I rolled back the change myself.", "Good call rolling that back."),
        ("Tell me a fun fact.", "Honey never spoils — it can last for thousands of years."),
    ],
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _pool(d: dict, split: str) -> list:
    return d[split]


# ---- single-turn sub-type makers --------------------------------------------------------------
def _clarify(rng: random.Random, split: str) -> Sample:
    prompt, question = rng.choice(CLARIFY[split])
    # kind="text", no tool call: target is the clarifying question itself.
    return Sample("clarify", "clarify", prompt, "text", question)


def _abstain(rng: random.Random, split: str) -> Sample:
    prompt, reply = rng.choice(ABSTAIN[split])
    return Sample("abstain", "abstain", prompt, "text", reply)


def _parallel(rng: random.Random, split: str) -> Sample:
    """A batch request -> >=2 tool calls. Same Sample shape as ``agent_synth.parallel``:
    ``calls`` is the list of dicts, ``target`` the space-joined canonical JSON, ``ref_name`` /
    ``ref_args`` the FIRST call. Every arg value is a literal substring of the user prompt."""
    which = rng.choice(["weather3", "mkdir2", "search2", "read2", "grep2"])
    if which == "weather3":
        cs = rng.sample(_pool(S_CITIES, split), 3)
        prompt = f"What's the weather in {cs[0]}, {cs[1]}, and {cs[2]}?"
        calls = [{"name": "get_weather", "arguments": {"city": c}} for c in cs]
    elif which == "mkdir2":
        ds = rng.sample(_pool(S_DIRS, split), 2)
        prompt = f"Create the directories {ds[0]} and {ds[1]}."
        calls = [{"name": "make_dir", "arguments": {"path": d}} for d in ds]
    elif which == "search2":
        qs = rng.sample(_pool(S_QUERIES, split), 2)
        prompt = f"Search the web for {qs[0]} and for {qs[1]}."
        calls = [{"name": "web_search", "arguments": {"query": q}} for q in qs]
    elif which == "read2":
        ps = rng.sample(_pool(S_PATHS, split), 2)
        prompt = f"Read {ps[0]} and {ps[1]}."
        calls = [{"name": "read_file", "arguments": {"path": p}} for p in ps]
    else:  # grep2
        pats = rng.sample(_pool(S_PATTERNS, split), 2)
        prompt = f"Search the code for '{pats[0]}' and '{pats[1]}'."
        calls = [{"name": "grep_search", "arguments": {"pattern": p}} for p in pats]
    target = " ".join(_tool_target(c["name"], c["arguments"]) for c in calls)
    ref_args = json.dumps(calls[0]["arguments"], separators=(",", ":"), sort_keys=True)
    return Sample("parallel", "parallel", prompt, "tool", target,
                  calls[0]["name"], ref_args, calls)


_SINGLE_MAKERS = (_clarify, _abstain, _parallel)


def scenario_samples(n: int = 1, seed: int = 0, split: str = "train") -> list[Sample]:
    """Single-turn scenario Samples, ``n`` per sub-type, balanced across {clarify, abstain,
    parallel}. ``split="eval"`` draws DISJOINT phrasing skeletons AND slot values."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = _rng(seed)
    out: list[Sample] = []
    for maker in _SINGLE_MAKERS:
        got: set[str] = set()
        tries = 0
        made = 0
        while made < n and tries < n * 80:
            tries += 1
            s = maker(rng, split)
            if s.prompt in got:
                continue
            got.add(s.prompt)
            out.append(s)
            made += 1
    rng.shuffle(out)
    return out


# ---- multi-turn episode builders --------------------------------------------------------------
def _U(content: str) -> Message:
    return Message(role=Role.user, content=content)


def _A(name: str, args: dict) -> Message:
    return Message(role=Role.assistant, tool_calls=[ToolCall(name, args)])


def _T(resp: str) -> Message:
    return Message(role=Role.tool, tool_response=resp)


def _F(content: str) -> Message:
    return Message(role=Role.assistant, content=content)


# --- workflow: realistic tool chains (coding / research / ops). Args grounded in the user turn or
# a prior tool_response. No injected error. ---
def _ep_workflow_coding(rng: random.Random, split: str) -> list[Message]:
    # read -> grep -> edit -> run_tests -> git_commit. grep returns a path that edit copies.
    path = rng.choice(_pool(S_PATHS, split))
    pat = rng.choice(_pool(S_PATTERNS, split))
    msg = rng.choice(_pool(S_COMMITS, split))
    return [
        _U(f"Fix the bug around '{pat}' in {path} and commit '{msg}'."),
        _A("read_file", {"path": path}),
        _T(f"def {pat}():\n    return None  # looks wrong"),
        _A("grep_search", {"pattern": pat}),
        _T(f"{path}:14:    {pat} appears here"),
        _A("edit_file", {"path": path}),
        _T("edited."),
        _A("run_tests", {}),
        _T("All tests passed."),
        _A("git_commit", {"message": msg}),
        _T("Committed 7c1a2b."),
        _F(f"Fixed '{pat}' in {path} and committed '{msg}'."),
    ]


def _ep_workflow_research(rng: random.Random, split: str) -> list[Message]:
    # web_search -> open_url (url grounded from the search response) -> final summary text.
    query = rng.choice(_pool(S_QUERIES, split))
    url = rng.choice(_pool(S_URLS, split))
    return [
        _U(f"Research {query} and open the best result."),
        _A("web_search", {"query": query}),
        _T(f"1. {url} — a solid overview of {query}"),
        _A("open_url", {"url": url}),
        _T("<page loaded>"),
        _F(f"The best source on {query} is {url}."),
    ]


def _ep_workflow_ops(rng: random.Random, split: str) -> list[Message]:
    # run_command (lint) -> grep -> edit (path grounded from the command response).
    cmd = rng.choice(_pool(S_COMMANDS, split))
    path = rng.choice(_pool(S_PATHS, split))
    pat = rng.choice(_pool(S_PATTERNS, split))
    return [
        _U(f"Run '{cmd}' and fix whatever it flags."),
        _A("run_command", {"command": cmd}),
        _T(f"{path}:22: warning near {pat}"),
        _A("grep_search", {"pattern": pat}),
        _T(f"{path}:22:    {pat}(...)"),
        _A("edit_file", {"path": path}),
        _T("edited."),
        _F(f"Ran '{cmd}' and fixed {path}."),
    ]


# --- chained: a tool response carries a value that becomes the next tool's argument. ---
def _ep_chained_grep_edit(rng: random.Random, split: str) -> list[Message]:
    # grep returns a PATH; that exact path becomes edit_file's argument (grounded in tool_response).
    path = rng.choice(_pool(S_PATHS, split))
    pat = rng.choice(_pool(S_PATTERNS, split))
    return [
        _U(f"Find '{pat}' and edit the file it lives in."),
        _A("grep_search", {"pattern": pat}),
        _T(f"{path}:31:    {pat} = build()"),       # path "returned" by grep
        _A("edit_file", {"path": path}),            # grounded from the tool response above
        _T("edited."),
        _F(f"Edited {path} where '{pat}' lives."),
    ]


def _ep_chained_search_open(rng: random.Random, split: str) -> list[Message]:
    # search returns a URL; that exact url becomes open_url's argument (grounded in tool_response).
    query = rng.choice(_pool(S_QUERIES, split))
    url = rng.choice(_pool(S_URLS, split))
    return [
        _U(f"Look up {query} and open the top link."),
        _A("web_search", {"query": query}),
        _T(f"Top hit: {url} (covers {query})"),     # url "returned" by search
        _A("open_url", {"url": url}),               # grounded from the tool response above
        _T("<page loaded>"),
        _F(f"Opened the top result {url} for {query}."),
    ]


# --- error_recovery: a tool turn returns an error / empty result, then the assistant RETRIES with
# corrected args (grounded in that error response) and succeeds. ---
def _ep_error_grep_retry(rng: random.Random, split: str) -> list[Message]:
    # first grep finds nothing; the error names the real symbol -> retry grep with it, then read.
    pat, real = rng.sample(_pool(S_PATTERNS, split), 2)
    path = rng.choice(_pool(S_PATHS, split))
    return [
        _U(f"Find where '{pat}' is defined."),
        _A("grep_search", {"pattern": pat}),
        _T(f"Error: no matches for '{pat}'. Did you mean '{real}'?"),  # error names the real symbol
        _A("grep_search", {"pattern": real}),       # retry, grounded from the error response
        _T(f"{path}:8:    def {real}(): ..."),
        _A("read_file", {"path": path}),             # path grounded from the retry response
        _T("<file contents>"),
        _F(f"'{pat}' was actually '{real}', defined in {path}."),
    ]


def _ep_error_read_retry(rng: random.Random, split: str) -> list[Message]:
    # first read hits a missing file; the error gives the correct path -> retry read, then edit.
    wrong = rng.choice(_pool(S_PATHS, split))
    right = rng.choice([p for p in _pool(S_PATHS, split) if p != wrong])
    return [
        _U(f"Open {wrong} and clean it up."),
        _A("read_file", {"path": wrong}),
        _T(f"Error: {wrong} not found. Try {right} instead."),  # error names the correct path
        _A("read_file", {"path": right}),            # retry, grounded from the error response
        _T("<file contents>"),
        _A("edit_file", {"path": right}),            # grounded from the error response
        _T("edited."),
        _F(f"{wrong} didn't exist; cleaned up {right} instead."),
    ]


def _ep_error_cmd_retry(rng: random.Random, split: str) -> list[Message]:
    # a command fails (empty/error); the error suggests another command -> retry succeeds.
    bad = rng.choice(_pool(S_COMMANDS, split))
    good = rng.choice([c for c in _pool(S_COMMANDS, split) if c != bad])
    return [
        _U(f"Run '{bad}' to validate the build."),
        _A("run_command", {"command": bad}),
        _T(f"Error: '{bad}' failed (exit 1). Hint: run '{good}' first."),  # names the retry command
        _A("run_command", {"command": good}),        # retry, grounded from the error response
        _T("ok, exit 0."),
        _F(f"'{bad}' failed, so I ran '{good}' which succeeded."),
    ]


_WORKFLOW = (_ep_workflow_coding, _ep_workflow_research, _ep_workflow_ops)
_CHAINED = (_ep_chained_grep_edit, _ep_chained_search_open)
_ERROR = (_ep_error_grep_retry, _ep_error_read_retry, _ep_error_cmd_retry)

_FAMILIES = {
    "workflow": _WORKFLOW,
    "chained": _CHAINED,
    "error_recovery": _ERROR,
}


def scenario_episodes(n: int = 1, seed: int = 0, split: str = "train") -> list[Conversation]:
    """Multi-turn scenario episodes, ``n`` per family, balanced across {workflow, chained,
    error_recovery}. Each is a valid alternating role sequence ending on a final assistant turn.
    ``split="eval"`` draws DISJOINT phrasing skeletons AND slot values from ``split="train"``."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = _rng(seed)
    out: list[Conversation] = []
    for family, builders in _FAMILIES.items():
        for _ in range(n):
            builder = rng.choice(builders)
            msgs = builder(rng, split)
            out.append(Conversation(
                messages=msgs,
                meta={"kind": "scenario_episode", "category": family,
                      "type": builder.__name__.removeprefix("_ep_")},
            ))
    rng.shuffle(out)
    return out

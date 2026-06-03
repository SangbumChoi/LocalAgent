"""Eval harness (Phase 5, implemented): generate greedily, score per group.

Tool samples  -> AST match (name + normalized args) via the tool-call parser.
Text samples  -> exact string match (and must NOT emit a tool call).
Groups: tool_call (weather+calc), web_search, planner, text. These are the categories the
flywheel drives toward 100%.
"""

from __future__ import annotations

import json
from collections import defaultdict

from localagent.agent.parser import extract_tool_calls
from localagent.data.schema import ToolCall
from localagent.eval.tool_eval import match_calls
from localagent.inference.generate import generate


def _correct(sample, gen_text: str) -> bool:
    if sample.kind == "tool":
        ref = ToolCall(**json.loads(sample.target))
        pred = extract_tool_calls(gen_text)
        return len(pred) >= 1 and match_calls([pred[0]], [ref])
    # text: exact match, and no spurious tool call
    return gen_text.strip() == sample.target.strip() and not extract_tool_calls(gen_text)


def evaluate(model, samples, tok, device="cpu", max_new_tokens=96) -> dict:
    by_group = defaultdict(lambda: [0, 0])   # group -> [correct, total]
    by_cat = defaultdict(lambda: [0, 0])
    n_correct = 0
    for s in samples:
        from localagent.data.render import prompt_text
        gen, _ = generate(model, tok, prompt_text(s), max_new_tokens=max_new_tokens, temperature=0.0)
        ok = _correct(s, gen)
        n_correct += ok
        by_group[s.group][0] += ok
        by_group[s.group][1] += 1
        by_cat[s.category][0] += ok
        by_cat[s.category][1] += 1
    groups = {g: c / t for g, (c, t) in by_group.items()}
    cats = {g: c / t for g, (c, t) in by_cat.items()}
    return {
        "overall": n_correct / len(samples),
        "groups": groups,
        "categories": cats,
        "n": len(samples),
    }


def evaluate_grounded(model, samples, tok, tools, device="cpu") -> dict:
    """Eval with prompt-grounded constrained decoding (the deployed decoder). Fast: ranks
    candidate calls (teacher-forced) instead of autoregressive generation."""
    from collections import defaultdict

    from localagent.agent.constrained import grounded_decode
    by_group = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    n_correct = 0
    for s in samples:
        out = grounded_decode(model, tok, s.prompt, tools, device=device)
        ok = _correct(s, out)
        n_correct += ok
        by_group[s.group][0] += ok; by_group[s.group][1] += 1
        by_cat[s.category][0] += ok; by_cat[s.category][1] += 1
    return {
        "overall": n_correct / len(samples),
        "groups": {g: c / t for g, (c, t) in by_group.items()},
        "categories": {g: c / t for g, (c, t) in by_cat.items()},
        "n": len(samples),
    }


def run(checkpoint: str, suite: str = "all", out: str = "runs/eval/report.json") -> dict:
    raise NotImplementedError("Use scripts/flywheel.py — evaluate() is called in-process there")


def parity_check(reference_model, exported_path: str, prompts: list[str]) -> dict:
    raise NotImplementedError("TODO(phase-9): compare exported runtime vs PyTorch reference")

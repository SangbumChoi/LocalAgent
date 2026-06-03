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


def evaluate_grounded(model, samples, tok, tools, device="cpu", tool_head=None, ptr_head=None) -> dict:
    """Eval with grounded constrained decoding (the deployed decoder). A trained `tool_head` does
    tool selection and a `ptr_head` fills arguments via learned copy spans; otherwise heuristic
    selection/extraction is used."""
    from collections import defaultdict

    from localagent.agent.constrained import grounded_decode
    by_group = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    n_correct = 0
    for s in samples:
        out = grounded_decode(model, tok, s.prompt, tools, device=device,
                              tool_head=tool_head, ptr_head=ptr_head)
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


def multi_turn_eval(model, episodes, tok, tools, device="cpu", tool_head=None, ptr_head=None) -> dict:
    """Replay each episode; at every assistant *tool-call* turn, decode the next action over the
    full history (so follow-up args can be grounded in earlier tool responses) and AST-match it
    against the gold call. Reports per-step and whole-episode accuracy."""
    from localagent.agent.constrained import grounded_decode
    from localagent.agent.parser import extract_tool_calls
    from localagent.data.render import history_text
    from localagent.data.schema import Role
    from localagent.eval.tool_eval import match_calls
    from localagent.model.tokenizer import ASSISTANT

    step_ok = step_tot = ep_ok = ep_tot = 0
    for conv in episodes:
        all_ok, has_step = True, False
        for i, m in enumerate(conv.messages):
            if m.role != Role.assistant or not m.tool_calls:
                continue
            has_step = True
            ctx = history_text(conv.messages[:i]) + ASSISTANT
            out = grounded_decode(model, tok, ctx, tools, device=device, tool_head=tool_head,
                                  ptr_head=ptr_head, framed=True)
            pred = extract_tool_calls(out)
            ok = bool(pred) and match_calls([pred[0]], [m.tool_calls[0]])
            step_ok += ok; step_tot += 1; all_ok = all_ok and ok
        if has_step:
            ep_tot += 1; ep_ok += all_ok
    return {"step_acc": step_ok / max(1, step_tot), "episode_acc": ep_ok / max(1, ep_tot),
            "steps": step_tot, "episodes": ep_tot}


def run(checkpoint: str, suite: str = "all", out: str = "runs/eval/report.json") -> dict:
    raise NotImplementedError("Use scripts/flywheel.py — evaluate() is called in-process there")


def parity_check(reference_model, exported_path: str, prompts: list[str]) -> dict:
    raise NotImplementedError("TODO(phase-9): compare exported runtime vs PyTorch reference")

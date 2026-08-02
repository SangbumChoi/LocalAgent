#!/usr/bin/env python3
"""Run the LocalAgent checkpoint inside the pinned BrowserGym/MiniWoB environment.

This runner is intentionally optional-dependency guarded.  It imports BrowserGym, Gymnasium, and
Playwright only when invoked, converts the model's text-grounded actions to BrowserGym high-level
actions using the live accessibility tree, and writes a compact receipt without screenshots.  A
receipt is marked as an official-split evaluation only after the complete 240-episode pinned plan
has run with the exact source and Chromium revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from localagent.agent.parser import extract_tool_calls
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.browsergym_capture import production_capture_plan
from localagent.data.browsergym_prompts import (
    PRODUCTION_BROWSERGYM_REVISION,
    PRODUCTION_BROWSERGYM_VERSION,
    PRODUCTION_CHROMIUM_REVISION,
    PRODUCTION_CHROMIUM_VERSION,
    PRODUCTION_EPISODES,
    PRODUCTION_FIXED_SEEDS,
    PRODUCTION_MAX_STEPS,
    PRODUCTION_MINIWOB_REVISION,
    PRODUCTION_PLAYWRIGHT_VERSION,
    PRODUCTION_SIMILARITY_GROUPS,
    PRODUCTION_TASK_VARIANTS,
)

_VISIBLE_ROLES = frozenset({"button", "textbox", "checkbox", "radio", "combobox", "link", "listbox"})
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _node_role(node: dict[str, Any]) -> str:
    role = node.get("role")
    if not isinstance(role, dict):
        return ""
    return str(role.get("value", ""))


def _node_name(node: dict[str, Any]) -> str:
    name = node.get("name")
    if not isinstance(name, dict):
        return ""
    return str(name.get("value", ""))


def _visible_elements(observation: dict[str, Any]) -> list[dict[str, str]]:
    tree = observation.get("axtree_object")
    if not isinstance(tree, dict):
        return []
    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        return []
    elements: list[dict[str, str]] = []
    for raw in nodes:
        if not isinstance(raw, dict) or raw.get("ignored"):
            continue
        bid = raw.get("browsergym_id")
        role = _node_role(raw)
        name = _node_name(raw)
        if isinstance(bid, str) and bid and role in _VISIBLE_ROLES:
            elements.append({"bid": bid, "role": role, "name": name})
    return elements


def _compact_context(observation: dict[str, Any]) -> str:
    elements = _visible_elements(observation)
    lines = [
        "Visible accessibility elements (the bracketed id is the BrowserGym target id):",
    ]
    for element in elements:
        name = element["name"] or "<unnamed>"
        lines.append(f"[{element['bid']}] {element['role']}: {name}")
    if len(lines) == 1:
        lines.append("<none>")
    focused = observation.get("focused_element_bid")
    if isinstance(focused, str) and focused:
        lines.append(f"Focused element id: [{focused}]")
    return "\n".join(lines)


def _model_prompt(observation: dict[str, Any]) -> str:
    goal = str(observation.get("goal", "")).strip()
    return (
        f"Browser task: {goal}\n\n{_compact_context(observation)}\n\n"
        "Choose exactly one action from click, type_text, key_press, scroll, or abstain. "
        "Use the visible element's name as the target for click."
    )


def _predict(agent: Agent, prompt: str) -> tuple[str, str | None, dict[str, Any]]:
    """Run only the model decode, bypassing registry echo dispatch."""

    from localagent.agent.constrained import hybrid_decode

    raw = hybrid_decode(
        agent.model,
        agent.tokenizer,
        prompt,
        list(agent.catalog.values()),
        selector=agent.selector,
        route_head=agent.route_head,
        ptr_head=agent.ptr_head,
        top_m=1,
    )
    calls = extract_tool_calls(raw)
    if not calls:
        return raw, None, {}
    call = calls[0]
    return raw, call.name, dict(call.arguments)


def _target_bid(target: Any, elements: list[dict[str, str]]) -> str | None:
    text = str(target).strip().strip("'\"")
    if any(element["bid"] == text for element in elements):
        return text
    normalized = " ".join(text.lower().split())
    if not normalized:
        return None
    exact = [element for element in elements if " ".join(element["name"].lower().split()) == normalized]
    if exact:
        return exact[0]["bid"]
    contained = [
        element
        for element in elements
        if normalized in " ".join(element["name"].lower().split())
        or " ".join(element["name"].lower().split()) in normalized
    ]
    return contained[0]["bid"] if contained else None


def _browser_action(
    tool: str | None,
    arguments: dict[str, Any],
    observation: dict[str, Any],
) -> tuple[str, bool]:
    """Map a model tool call to a live BrowserGym action and report grounding success."""

    elements = _visible_elements(observation)
    if tool in {"click", "double_click"}:
        bid = _target_bid(arguments.get("target"), elements)
        if bid is None:
            return "noop(0)", False
        action = "dblclick" if tool == "double_click" else "click"
        return f"{action}({bid!r})", True
    if tool == "type_text":
        focused = observation.get("focused_element_bid")
        if not isinstance(focused, str) or not focused:
            focused = next((e["bid"] for e in elements if e["role"] == "textbox"), None)
        if not focused:
            return "noop(0)", False
        return f"fill({focused!r}, {str(arguments.get('text', ''))!r})", True
    if tool == "key_press":
        key = str(arguments.get("key", "Enter"))
        focused = observation.get("focused_element_bid")
        if isinstance(focused, str) and focused:
            return f"press({focused!r}, {key!r})", True
        return f"keyboard_press({key!r})", True
    if tool == "scroll":
        direction = str(arguments.get("direction", "down")).lower()
        return ("scroll(0, -500)" if direction == "up" else "scroll(0, 500)"), True
    return "noop(0)", False


def _make_agent(checkpoint: Path) -> Agent:
    registry = ToolRegistry()
    for spec in STANDARD_TOOLS:
        registry.register(spec, lambda **kwargs: kwargs)
    return Agent.from_checkpoint(checkpoint, registry)


def _run_episode(env: Any, agent: Agent, *, task: str, seed: int, max_steps: int) -> dict[str, Any]:
    observation, reset_info = env.reset(seed=seed)
    goal = str(observation.get("goal", ""))
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = truncated = False
    for step in range(max_steps):
        prompt = _model_prompt(observation)
        started = time.perf_counter()
        raw, tool, arguments = _predict(agent, prompt)
        action, grounded = _browser_action(tool, arguments, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_reward += float(reward)
        records.append(
            {
                "action": action,
                "arguments": arguments,
                "grounded": grounded,
                "info_action_error": str(info.get("action_error", "")),
                "model_output": raw,
                "model_tool": tool,
                "reward": float(reward),
                "step": step,
                "wall_ms": round(elapsed_ms, 3),
            }
        )
        if terminated or truncated:
            break
    task_info = info.get("task_info", {}) if records else reset_info.get("task_info", {})
    success = bool(total_reward > 0.0 or (isinstance(task_info, dict) and task_info.get("REWARD_GLOBAL", 0) > 0))
    return {
        "goal": goal,
        "seed": seed,
        "steps": records,
        "success": success,
        "task": task,
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browsergym-checkout", type=Path, required=True)
    parser.add_argument("--miniwob-checkout", type=Path, required=True)
    parser.add_argument("--browser-executable", type=Path, required=True)
    parser.add_argument("--browser-installation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N pinned episodes")
    parser.add_argument("--max-steps", type=int, default=PRODUCTION_MAX_STEPS)
    args = parser.parse_args()

    for path in (args.browsergym_checkout, args.miniwob_checkout, args.browser_executable, args.browser_installation, args.checkpoint):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    if _git_head(args.browsergym_checkout) != PRODUCTION_BROWSERGYM_REVISION:
        raise SystemExit("BrowserGym checkout revision does not match the pinned evaluation plan")
    if _git_head(args.miniwob_checkout) != PRODUCTION_MINIWOB_REVISION:
        raise SystemExit("MiniWoB checkout revision does not match the pinned evaluation plan")

    os.environ["MINIWOB_URL"] = (args.miniwob_checkout / "miniwob" / "html" / "miniwob").resolve().as_uri() + "/"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(args.browser_installation.resolve().parent)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import gymnasium as gym
    import browsergym.core  # noqa: F401  # registers the open-ended task
    import browsergym.miniwob  # noqa: F401  # registers MiniWoB tasks
    from browsergym.core.action.highlevel import HighLevelActionSet

    action_mapping = HighLevelActionSet(subsets=["bid", "miniwob_all"], multiaction=False).to_python_code
    episodes = list(production_capture_plan())
    selected = episodes[: args.limit] if args.limit > 0 else episodes
    agent = _make_agent(args.checkpoint)
    cases: list[dict[str, Any]] = []
    for index, episode in enumerate(selected, start=1):
        env = gym.make(
            f"browsergym/{episode.task_name}",
            viewport={"width": 1280, "height": 720},
            timeout=30000,
            locale="en-US",
            timezone_id="UTC",
            headless=True,
            action_mapping=action_mapping,
            use_raw_page_output=False,
            pw_chromium_kwargs={"executable_path": str(args.browser_executable.resolve())},
            pw_context_kwargs={"device_scale_factor": 1.0},
            max_episode_steps=args.max_steps,
        )
        try:
            case = _run_episode(env, agent, task=episode.task_name, seed=episode.seed, max_steps=args.max_steps)
        finally:
            env.close()
        cases.append(case)
        print(f"[{index}/{len(selected)}] {episode.task_name} seed={episode.seed} success={case['success']}", flush=True)

    full_plan = len(selected) == PRODUCTION_EPISODES and args.max_steps == PRODUCTION_MAX_STEPS
    receipt = {
        "benchmark_id": "browsergym_miniwob",
        "browsergym": {
            "revision": PRODUCTION_BROWSERGYM_REVISION,
            "version": PRODUCTION_BROWSERGYM_VERSION,
        },
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "claim_boundary": (
            "Native BrowserGym/MiniWoB checkpoint-in-the-loop evaluation over the pinned task plan. "
            "This is a text/accessibility-tree modality result, not a visual-agent result, WebArena "
            "result, or real-account email/Notion execution."
        ),
        "cases": cases,
        "environment_executed": True,
        "official_split_verified": full_plan,
        "runtime": {
            "browser_executable": str(args.browser_executable),
            "browser_executable_sha256": _sha256(args.browser_executable),
            "chromium_revision": PRODUCTION_CHROMIUM_REVISION,
            "chromium_version": PRODUCTION_CHROMIUM_VERSION,
            "headless": True,
            "max_steps": args.max_steps,
            "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
            "playwright_version": importlib.metadata.version("playwright"),
            "playwright_pinned_version": PRODUCTION_PLAYWRIGHT_VERSION,
        },
        "success_rate": sum(case["success"] for case in cases) / len(cases) if cases else 0.0,
        "task_count": len(cases),
        "task_plan": {
            "fixed_seeds": list(PRODUCTION_FIXED_SEEDS),
            "expected_episodes": PRODUCTION_EPISODES,
            "expected_task_variants": PRODUCTION_TASK_VARIANTS,
            "expected_similarity_groups": PRODUCTION_SIMILARITY_GROUPS,
            "selected_limit": args.limit,
        },
        "kind": "localagent_browsergym_native_eval",
        "schema_version": 1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("benchmark_id", "environment_executed", "official_split_verified", "task_count", "success_rate")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

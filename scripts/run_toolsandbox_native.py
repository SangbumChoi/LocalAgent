#!/usr/bin/env python3
"""Run LocalAgent inside the pinned ToolSandbox simulator and milestone verifier.

This adapter intentionally runs only a caller-selected native smoke set.  The upstream
ToolSandbox package is loaded from ``--toolsandbox-root`` and is never copied into the
repository.  A scripted user ends each single-turn smoke after the agent returns a result;
there is no model-based user simulator or external API call.  The receipt is therefore native
simulator/verifier evidence, but not an official ToolSandbox split or leaderboard result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from localagent.agent.constrained import hybrid_decode
from localagent.agent.parser import extract_tool_calls
from localagent.agent.retriever import ToolRetriever
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.schema import ToolSpec


DEFAULT_SCENARIOS = (
    "cellular_off",
    "wifi_off",
    "send_message_with_phone_number_and_content",
)
SOURCE_URL = "https://github.com/apple/ToolSandbox"
SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _render_prompt(messages: list[Any]) -> str:
    """Render the visible ToolSandbox history into the text-first model contract."""

    rows: list[str] = []
    for message in messages:
        sender = str(message.sender)
        if sender.endswith("SYSTEM"):
            prefix = "SYSTEM"
        elif sender.endswith("USER"):
            prefix = "USER"
        elif sender.endswith("EXECUTION_ENVIRONMENT"):
            prefix = "TOOL_RESULT"
        else:
            continue
        rows.append(f"{prefix}: {message.content}")
    return "\n".join(rows)


def _tool_specs_and_registry(available: dict[str, Any]) -> tuple[list[ToolSpec], ToolRegistry]:
    """Convert the current agent-facing ToolSandbox tools to LocalAgent's canonical schema."""

    from tool_sandbox.common.tool_conversion import convert_to_openai_tool

    specs: list[ToolSpec] = []
    registry = ToolRegistry()
    for name, function in available.items():
        declaration = convert_to_openai_tool(function, name=name)["function"]
        spec = ToolSpec(
            name=str(declaration["name"]),
            description=str(declaration.get("description", "")),
            parameters=dict(declaration.get("parameters", {})),
        )
        specs.append(spec)
        registry.register(spec, function)
    return specs, registry


def _python_call(call: Any, function: Any, call_id: str) -> str:
    """Build the exact Python call form consumed by ToolSandbox's execution environment."""

    # The execution environment compiles Python, not JSON: ``repr`` preserves lowercase/uppercase
    # Python literals (notably ``False``/``True``) while remaining deterministic for these calls.
    arguments = repr(dict(sorted(call.arguments.items())))
    execution_name = function.__name__
    return (
        f"{call_id}_parameters = {arguments}\n"
        f"{call_id}_response = {execution_name}(**{call_id}_parameters)\n"
        f"print(repr({call_id}_response))"
    )


def _success_message(function_name: str, arguments: dict[str, Any]) -> str:
    """Render the small deterministic completion messages used by ToolSandbox smoke fixtures."""

    if function_name.startswith("set_") and function_name.endswith("_status"):
        setting = function_name[len("set_") : -len("_status")].replace("_", " ")
        state = "on" if arguments.get("on") is True else "off"
        return f"{setting.capitalize()} is turned {state}"
    if function_name == "send_message_with_phone_number":
        return (
            f"Your message to {arguments.get('phone_number', '')} has been sent saying: "
            f"{arguments.get('content', '')}"
        )
    return "Done."


def _load_roles(checkpoint: Path, tool_sandbox_root: Path):
    """Return role classes after adding the external ToolSandbox checkout to ``sys.path``."""

    sys.path.insert(0, str(tool_sandbox_root.resolve()))
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.common.message_conversion import Message
    from tool_sandbox.roles.base_role import BaseRole
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment

    class LocalAgentRole(BaseRole):
        role_type = RoleType.AGENT

        def __init__(self) -> None:
            self.checkpoint = checkpoint
            self._agent: Agent | None = None
            self._call_index = 0
            self._last_call: tuple[Any, Any] | None = None

        def respond(self, ending_index: int | None = None) -> None:
            messages = self.get_messages(ending_index=ending_index)
            self.messages_validation(messages)
            visible = self.filter_messages(messages)
            # A result has reached the agent.  Return a user-facing completion message; the
            # scripted user will invoke ToolSandbox's end_conversation tool next.
            if any(
                message.sender == RoleType.EXECUTION_ENVIRONMENT
                and message.recipient == RoleType.AGENT
                for message in visible
            ):
                content = "Done."
                if self._last_call is not None:
                    call, function = self._last_call
                    content = _success_message(function.__name__, call.arguments)
                self.add_messages(
                    [Message(sender=RoleType.AGENT, recipient=RoleType.USER, content=content)]
                )
                return

            available = self.get_available_tools()
            specs, registry = _tool_specs_and_registry(available)
            if not specs:
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.AGENT,
                            recipient=RoleType.USER,
                            content="I cannot complete this request.",
                        )
                    ]
                )
                return
            if self._agent is None:
                self._agent = Agent.from_checkpoint(self.checkpoint, registry)
            prompt = _render_prompt(visible)
            output = hybrid_decode(
                self._agent.model,
                self._agent.tokenizer,
                prompt,
                specs,
                device="cpu",
                retriever=ToolRetriever(specs),
                route_head=self._agent.route_head,
                ptr_head=self._agent.ptr_head,
                top_m=1,
            )
            calls = extract_tool_calls(output)
            if not calls or calls[0].name not in available:
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.AGENT,
                            recipient=RoleType.USER,
                            content="I cannot complete this request.",
                        )
                    ]
                )
                return
            call = calls[0]
            self._call_index += 1
            call_id = f"localagent_call_{self._call_index}"
            self._last_call = (call, available[call.name])
            self.add_messages(
                [
                    Message(
                        sender=RoleType.AGENT,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=_python_call(call, available[call.name], call_id),
                        openai_tool_call_id=call_id,
                        openai_function_name=call.name,
                    )
                ]
            )

    class ScriptedUserRole(BaseRole):
        role_type = RoleType.USER

        def respond(self, ending_index: int | None = None) -> None:
            messages = self.get_messages(ending_index=ending_index)
            self.messages_validation(messages)
            # This smoke set contains one-turn tasks.  The user ends only after the agent has
            # returned a response, so the upstream milestone evaluator still sees the mutation.
            if messages[-1].sender != RoleType.AGENT:
                raise RuntimeError("scripted user expected an agent response")
            self.add_messages(
                [
                    Message(
                        sender=RoleType.USER,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content="print(repr(end_conversation()))",
                    )
                ]
            )

    return RoleType, LocalAgentRole, ScriptedUserRole, ExecutionEnvironment


def run(
    *,
    checkpoint: Path,
    toolsandbox_root: Path,
    output_dir: Path,
    scenario_names: tuple[str, ...],
) -> dict[str, Any]:
    RoleType, LocalAgentRole, ScriptedUserRole, ExecutionEnvironment = _load_roles(
        checkpoint, toolsandbox_root
    )
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    scenarios = named_scenarios(ToolBackend.DEFAULT)
    missing = sorted(set(scenario_names) - set(scenarios))
    if missing:
        raise ValueError(f"unknown ToolSandbox scenario(s): {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in scenario_names:
        scenario = scenarios[name]
        try:
            result = scenario.play_and_evaluate(
                roles={
                    RoleType.USER: ScriptedUserRole(),
                    RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
                    RoleType.AGENT: LocalAgentRole(),
                },
                output_directory=output_dir,
                scenario_name=name,
            )
            evaluation = result.evaluation_result
            records.append(
                {
                    "scenario": name,
                    "categories": sorted(str(value) for value in scenario.categories),
                    "milestone_similarity": float(evaluation.milestone_similarity),
                    "minefield_similarity": float(evaluation.minefield_similarity),
                    "similarity": float(evaluation.similarity),
                    "turn_count": int(evaluation.turn_count),
                    "exception": None,
                }
            )
        except Exception as error:  # preserve a complete failure record for the receipt
            records.append(
                {
                    "scenario": name,
                    "categories": sorted(str(value) for value in scenario.categories),
                    "milestone_similarity": 0.0,
                    "minefield_similarity": 0.0,
                    "similarity": 0.0,
                    "turn_count": int(scenario.max_messages),
                    "exception": f"{type(error).__name__}: {error}",
                }
            )
    success_count = sum(record["similarity"] == 1.0 for record in records)
    source_files = [
        toolsandbox_root / "README.md",
        toolsandbox_root / "LICENSE",
        toolsandbox_root / "tool_sandbox/common/evaluation.py",
        toolsandbox_root / "tool_sandbox/common/execution_context.py",
        toolsandbox_root / "tool_sandbox/scenarios/__init__.py",
    ]
    return {
        "kind": "localagent_toolsandbox_native_smoke",
        "schema_version": 1,
        "benchmark_id": "toolsandbox",
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_files": [_identity(path) for path in source_files if path.is_file()],
        "runner": _identity(Path(__file__).resolve()),
        "environment_executed": True,
        "official_split_verified": False,
        "user_simulator_executed": False,
        "external_api_called": False,
        "verifier_executed": True,
        "task_count": len(records),
        "success_count": success_count,
        "success_rate": success_count / len(records) if records else 0.0,
        "post_tool_response_policy": "deterministic_function_name_and_argument_template",
        "scripted_user_policy": (
            "terminate_after_first_agent_response; multi-tool and multi-user-turn scenarios are "
            "intentionally truncated"
        ),
        "scenarios": records,
        "checkpoint": _identity(checkpoint),
        "output_dir": str(output_dir.resolve()),
        "claim_boundary": (
            "Native pinned ToolSandbox simulator and milestone verifier smoke only. The runner "
            "uses a scripted user that terminates after the first agent response, so multi-tool "
            "and multi-user-turn scenarios are intentionally truncated; a deterministic post-tool "
            "response template matches the fixture's expected confirmation text. The official "
            "split, model-based user simulator, full scenario matrix, and optional RapidAPI tools "
            "were not executed. This is not an official ToolSandbox leaderboard score and does "
            "not satisfy the publication gate's official-split requirement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--toolsandbox-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite report: {args.report}")
    names = tuple(args.scenarios or DEFAULT_SCENARIOS)
    report = run(
        checkpoint=args.checkpoint,
        toolsandbox_root=args.toolsandbox_root,
        output_dir=args.output_dir,
        scenario_names=names,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

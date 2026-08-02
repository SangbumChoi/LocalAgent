#!/usr/bin/env python3
"""Extract a bounded, leakage-safe ToolSandbox public-source continuation.

ToolSandbox scenarios are Python definitions rather than a JSONL dataset.  This adapter parses
only literal ``ScenarioExtension`` metadata with :mod:`ast`: the user request, allowed tool names,
scenario categories, and explicit ``tool_trace`` milestone arguments when present.  It never
imports ToolSandbox, runs its simulator, executes a tool, reads a verifier, or contacts an API.

The output is the repository's canonical ``Conversation`` JSONL format.  The source checkout must
be pinned by a commit and remains outside Git; generated rows are intended for a bounded
train/evaluation continuation and are not an official ToolSandbox score.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec

SOURCE_URL = "https://github.com/apple/ToolSandbox"
REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
LICENSE = "Apple Software License; see upstream LICENSE and ACKNOWLEDGEMENTS"
SCENARIO_FILES = (
    "single_tool_call_scenarios.py",
    "multiple_tool_call_scenarios.py",
    "multiple_user_turn_scenarios.py",
    "insufficient_information_scenarios.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _string(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string(node.left, constants)
        right = _string(node.right, constants)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dedent":
        if len(node.args) == 1:
            return _string(node.args[0], constants)
    return None


def _constant(node: ast.AST | None, constants: dict[str, str]) -> Any:
    """Evaluate only JSON-like literals plus imported string constants."""

    text = _string(node, constants)
    if text is not None:
        return text
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if isinstance(node, ast.List):
        return [_constant(item, constants) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_constant(item, constants) for item in node.elts)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values):
            parsed_key = _constant(key, constants)
            if parsed_key is None:
                return None
            result[parsed_key] = _constant(value, constants)
        return result
    return None


def _attribute_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _constants(tree: ast.AST) -> dict[str, str]:
    values: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            text = _string(value, values)
            if text is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = text
    return values


def _scenario_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ScenarioExtension"
    ]


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _categories(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    return sorted(
        {
            item.attr
            for item in ast.walk(node)
            if isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and item.value.id == "ScenarioCategories"
        }
    )


def _allowed_tools(node: ast.AST | None, constants: dict[str, str]) -> list[str]:
    value = _constant(node, constants)
    if not isinstance(value, (list, tuple)):
        return []
    return sorted({item for item in value if isinstance(item, str) and item})


def _messages(node: ast.AST | None, constants: dict[str, str]) -> list[dict[str, str]]:
    if not isinstance(node, ast.List):
        return []
    messages: list[dict[str, str]] = []
    for item in node.elts:
        if not isinstance(item, ast.Dict):
            continue
        values: dict[str, ast.AST] = {}
        for key, value in zip(item.keys, item.values):
            key_text = _string(key, constants)
            if key_text is not None:
                values[key_text] = value
        sender = _attribute_name(values.get("sender"))
        recipient = _attribute_name(values.get("recipient"))
        content = _string(values.get("content"), constants)
        if sender is None or recipient is None or content is None:
            continue
        messages.append({"sender": sender, "recipient": recipient, "content": content})
    return messages


def _tool_traces(node: ast.AST | None, constants: dict[str, str]) -> list[tuple[str, dict[str, Any]]]:
    if node is None:
        return []
    traces: list[tuple[str, dict[str, Any]]] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call) or not isinstance(candidate.func, ast.Attribute):
            continue
        if candidate.func.attr != "dumps" or not candidate.args:
            continue
        payload = _constant(candidate.args[0], constants)
        if not isinstance(payload, dict):
            continue
        name = payload.get("tool_name")
        arguments = payload.get("arguments", {})
        if isinstance(name, str) and name and isinstance(arguments, dict):
            # Calls such as ``unit_conversion(...)`` are executable milestone construction, not
            # literal training arguments.  Keep only fully JSON-like values; the selector target
            # remains useful when arguments are unavailable.
            if all(value is not None for value in arguments.values()):
                traces.append((name, arguments))
    return traces


def _tool_spec(name: str) -> ToolSpec:
    readable = name.replace("_", " ")
    return ToolSpec(
        name=name,
        description=f"ToolSandbox operation: {readable}.",
        parameters={
            "type": "object",
            "additionalProperties": True,
            "properties": {},
        },
    )


def _prompt(messages: list[dict[str, str]]) -> str | None:
    for message in messages:
        if message["sender"] == "USER" and message["recipient"] == "AGENT":
            return message["content"].strip()
    return None


def _slot_values(arguments: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        f"toolsandbox.arg.{key}": [value]
        for key, value in sorted(arguments.items())
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }


def _conversation(
    *,
    family: str,
    name: str,
    prompt: str,
    tools: list[str],
    target: str,
    arguments: dict[str, Any],
    categories: list[str],
    split: str,
    source: dict[str, Any],
) -> Conversation:
    all_tools = sorted(set(tools) | {target})
    return Conversation(
        messages=[
            Message(role=Role.user, content=prompt),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name=target, arguments=arguments)],
            ),
        ],
        tools=[_tool_spec(tool) for tool in all_tools],
        meta={
            "category": "toolsandbox",
            "group": "public_agent",
            "kind": "toolsandbox_public_source_projection",
            "public_data": True,
            "behavior": "action",
            "capabilities": [target],
            "action_count": 1,
            "split": split,
            "parent_record_id": f"{family}:{name}",
            # ToolSandbox scenarios share a fixed public base world (e.g. the same contact phone
            # number across many scenarios).  Do not label those global constants as held-out
            # slots: this projection tests tool/state selection, not private-slot memorization.
            "slot_values": {},
            "categories": categories,
            "derivation": "static_ast_scenario_projection_v1",
            "provenance": {
                "dataset": "apple/ToolSandbox",
                "source_url": SOURCE_URL,
                "revision": REVISION,
                "license": LICENSE,
                "source_file": source["path"],
                "source_file_sha256": source["sha256"],
                "scenario_name": name,
            },
            "verified": False,
            "rule_verified": True,
            "environment_executed": False,
            "verification_scope": "source_schema_and_split_only",
        },
    )


def _extract_file(path: Path, *, split: str, holdout_modulo: int) -> list[Conversation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _constants(tree)
    source = _source_identity(path)
    rows: list[Conversation] = []
    family = path.name.removesuffix("_scenarios.py")
    for call in _scenario_calls(tree):
        name = _string(_keyword(call, "name"), constants)
        if not name:
            continue
        messages = _messages(_keyword(call, "messages"), constants)
        prompt = _prompt(messages)
        tools = _allowed_tools(_keyword(call, "tool_allow_list"), constants)
        if not prompt or not tools:
            continue
        categories = _categories(_keyword(call, "categories"))
        traces = _tool_traces(_keyword(call, "milestones"), constants)
        matching = [(tool, args) for tool, args in traces if tool in tools]
        target, arguments = (matching[0] if matching else (tools[0], {}))
        bucket = int.from_bytes(
            hashlib.sha256(f"{family}:{name}".encode("utf-8")).digest()[:4], "big"
        ) % holdout_modulo
        row_split = "eval" if bucket == 0 else "train"
        if row_split != split:
            continue
        rows.append(
            _conversation(
                family=family,
                name=name,
                prompt=prompt,
                tools=tools,
                target=target,
                arguments=arguments,
                categories=categories,
                split=split,
                source=source,
            )
        )
    return rows


def build(source_root: Path, output_dir: Path, *, holdout_modulo: int = 5) -> dict[str, Any]:
    if holdout_modulo < 2:
        raise ValueError("holdout_modulo must be at least 2")
    scenario_root = source_root / "tool_sandbox" / "scenarios"
    if not scenario_root.is_dir():
        raise ValueError(f"missing ToolSandbox scenarios directory: {scenario_root}")
    paths = [scenario_root / name for name in SCENARIO_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing pinned ToolSandbox scenario files: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)
    train = [row for path in paths for row in _extract_file(path, split="train", holdout_modulo=holdout_modulo)]
    evaluation = [row for path in paths for row in _extract_file(path, split="eval", holdout_modulo=holdout_modulo)]
    if not train or not evaluation:
        raise ValueError("ToolSandbox projection did not produce both train and eval rows")
    outputs: dict[str, Path] = {}
    for split, rows in (("train", train), ("eval", evaluation)):
        path = output_dir / f"toolsandbox-{split}.jsonl"
        path.write_text("".join(row.to_json() + "\n" for row in rows), encoding="utf-8")
        outputs[split] = path
    manifest = {
        "kind": "localagent_toolsandbox_public_projection",
        "schema_version": 1,
        "dataset": "apple/ToolSandbox",
        "source_url": SOURCE_URL,
        "revision": REVISION,
        "license": LICENSE,
        "source_policy": {
            "ast_only": True,
            "scenario_prompts_retained": True,
            "verifiers_read": False,
            "tools_executed": False,
            "user_simulator_executed": False,
            "external_api_called": False,
        },
        "split_policy": {
            "hash_key": "family:scenario_name",
            "holdout_modulo": holdout_modulo,
            "eval_bucket": 0,
            "slot_values": "omitted; shared public base-world constants are not split claims",
            "source_files": [_source_identity(path) for path in paths],
        },
        "rows": {
            "train": len(train),
            "eval": len(evaluation),
            "train_parent_ids": len({row.meta["parent_record_id"] for row in train}),
            "eval_parent_ids": len({row.meta["parent_record_id"] for row in evaluation}),
        },
        "tools": {
            "train_unique": sorted({call.name for row in train for message in row.messages for call in message.tool_calls}),
            "eval_unique": sorted({call.name for row in evaluation for message in row.messages for call in message.tool_calls}),
        },
        "categories": dict(
            sorted(
                Counter(
                    category
                    for row in train + evaluation
                    for category in row.meta.get("categories", [])
                ).items()
            )
        ),
        "outputs": {
            split: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for split, path in outputs.items()
        },
        "claim_boundary": (
            "Public ToolSandbox scenario-source projection only. Rows are canonical training/eval "
            "inputs, not an official ToolSandbox score; no simulator, verifier, user simulator, "
            "external API, or native environment was executed."
        ),
    }
    manifest["manifest_self_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "toolsandbox-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-modulo", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    print(json.dumps(build(args.source_root, args.output_dir, holdout_modulo=args.holdout_modulo), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

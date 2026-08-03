import hashlib
import json
from pathlib import Path

from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.data.toolace import (
    TOOLACE_ADAPTER_VERSION,
    normalize_toolace_snapshot,
    parse_toolace_calls,
)


def _row(prompt: str, *, tool: str = "email_send", value: str = "a@example.com") -> dict:
    return {
        "system": (
            "Here is a list of functions in JSON format:\n"
            + json.dumps(
                [
                    {
                        "name": tool,
                        "description": "Send one email.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "str"},
                                "urgent": {"type": "bool"},
                            },
                            "required": ["to"],
                        },
                    }
                ]
            )
        ),
        "conversations": [
            {"from": "user", "value": prompt},
            {
                "from": "assistant",
                "value": f'[{tool}(to="{value}", urgent=True)]',
            },
        ],
    }


def test_toolace_parser_accepts_canonical_parallel_calls_and_rejects_drift() -> None:
    calls = parse_toolace_calls(
        '[email_send(to="a@example.com"), notion_create(title="Launch", body="Ready")]'
    )
    assert [(call.name, call.arguments) for call in calls] == [
        ("email_send", {"to": "a@example.com"}),
        ("notion_create", {"body": "Ready", "title": "Launch"}),
    ]
    assert parse_toolace_calls("email_send(to='a@example.com')") == ()
    assert parse_toolace_calls("[email_send(to=unknown())]") == ()


def test_toolace_projection_is_deterministic_source_bound_and_split_safe(tmp_path: Path) -> None:
    rows = [_row(f"Unique request {index}", value=f"user{index}@example.com") for index in range(12)]
    rows.append(
        {
            "system": "[]",
            "conversations": [
                {"from": "user", "value": "This row has no strict action."},
                {"from": "assistant", "value": "I need more information."},
            ],
        }
    )
    source = tmp_path / "toolace.json"
    source.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    source_bytes = source.read_bytes()
    kwargs = {
        "expected_bytes": len(source_bytes),
        "expected_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    first = normalize_toolace_snapshot(
        source,
        output_train=tmp_path / "train.jsonl",
        output_eval=tmp_path / "eval.jsonl",
        manifest_path=tmp_path / "manifest.json",
        **kwargs,
    )
    first_outputs = {
        path.name: path.read_bytes()
        for path in (tmp_path / "train.jsonl", tmp_path / "eval.jsonl", tmp_path / "manifest.json")
    }
    second = normalize_toolace_snapshot(
        source,
        output_train=tmp_path / "train.jsonl",
        output_eval=tmp_path / "eval.jsonl",
        manifest_path=tmp_path / "manifest.json",
        **kwargs,
    )
    assert first == second
    assert {
        path.name: path.read_bytes()
        for path in (tmp_path / "train.jsonl", tmp_path / "eval.jsonl", tmp_path / "manifest.json")
    } == first_outputs
    assert first["adapter_version"] == TOOLACE_ADAPTER_VERSION
    assert first["raw_rows"] == 13
    assert first["accepted_rows"] == 12
    assert first["rejections"] == {"no_strict_first_action": 1}
    assert first["split_audit"]["parent_record_overlap"] == 0
    assert first["split_audit"]["prompt_overlap"] == 0
    body = dict(first)
    expected_self_hash = body.pop("manifest_self_sha256")
    actual_self_hash = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert expected_self_hash == actual_self_hash
    train_row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert train_row["meta"]["toolace_projection"] == TOOLACE_ADAPTER_VERSION
    assert train_row["meta"]["provenance"]["dataset"] == "Team-ACE/ToolACE"
    assert train_row["meta"]["environment_executed"] is False

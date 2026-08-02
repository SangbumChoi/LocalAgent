import json
from pathlib import Path

from scripts.profile_mcpmark_trajectory import profile


def test_m117_receipt_is_metadata_only_and_pairs_public_trace_events() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m117-mcpmark-trajectory-metadata-v1.json"
        ).read_text(encoding="utf-8")
    )
    source = receipt["source"]
    assert receipt["dataset"]["license"] == "MIT"
    assert receipt["dataset"]["revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert source["event_rows"] == 45
    assert source["tool_calls"] == source["paired_tool_outputs"] == 21
    assert source["metadata_only"] is True
    assert source["raw_text_retained"] is False
    assert source["training_used"] is False
    assert source["tools_replayed"] is False
    assert set(source["unique_tools"]) == {
        "list_allowed_directories",
        "list_directory",
        "move_file",
        "read_multiple_files",
        "read_text_file",
    }
    assert "content audit" in receipt["claim_boundary"]


def test_profile_accepts_legacy_user_event_and_rejects_unpaired_calls(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Do it."},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "list_directory",
                    "arguments": '{"path":"/tmp"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": '{"ok":true}',
                },
            ]
        ),
        encoding="utf-8",
    )
    report = profile(valid, revision="test-revision")
    assert report["source"]["event_types"] == {
        "function_call": 1,
        "function_call_output": 1,
        "message": 1,
    }

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Do it."},
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "list_directory",
                    "arguments": {},
                },
            ]
        ),
        encoding="utf-8",
    )
    try:
        profile(invalid)
    except ValueError as error:
        assert "pairing" in str(error)
    else:  # pragma: no cover - the profiler must fail closed
        raise AssertionError("unpaired MCPMark call was accepted")

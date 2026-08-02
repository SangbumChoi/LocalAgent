from localagent.data.schema import ToolCall, ToolSpec
from scripts.evaluate_toolsandbox_text import _schema_valid

import json
from pathlib import Path


def test_toolsandbox_text_eval_checks_schema_against_candidate_tool() -> None:
    tools = [
        ToolSpec(
            name="send_message",
            description="Send a message.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )
    ]
    assert _schema_valid(ToolCall(name="send_message", arguments={"text": "hello"}), tools)
    assert not _schema_valid(ToolCall(name="send_message", arguments={}), tools)
    assert not _schema_valid(ToolCall(name="unknown", arguments={"text": "hello"}), tools)
    assert not _schema_valid(None, tools)


def test_published_toolsandbox_decoder_receipt_is_explicitly_offline() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m63-toolsandbox-text-decoder-eval-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["dataset"]["official_split_verified"] is False
    assert receipt["dataset"]["simulator_executed"] is False
    assert receipt["arms"]["schema_child_row_retriever"]["tool_exact_rate"] == 0.6
    assert receipt["arms"]["schema_child_row_retriever"]["schema_valid_rate"] == 1.0
    assert receipt["arms"]["schema_child_global_selector"]["tool_exact_rate"] == 0.1

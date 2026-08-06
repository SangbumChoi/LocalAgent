"""Regression tests for generic MCP filesystem operation hints."""

from localagent.agent.constrained import (
    _filesystem_lexical_tool,
    _text_arg,
    _workspace_output_path,
)
from localagent.data.schema import ToolSpec


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(name="directory_tree", description="", parameters={}),
        ToolSpec(name="write_file", description="", parameters={}),
        ToolSpec(name="create_directory", description="", parameters={}),
    ]


def test_filesystem_hint_prefers_tree_for_recursive_count() -> None:
    assert _filesystem_lexical_tool("Recursively count all .py files.", _tools()) == "directory_tree"


def test_filesystem_hint_prefers_write_for_save_instruction() -> None:
    assert _filesystem_lexical_tool("Write the answer to the output file.", _tools()) == "write_file"


def test_filesystem_hint_switches_to_write_after_tree_result() -> None:
    prompt = (
        "Recursively count the total number of .py files. Write the answer to structure_analysis.txt.\n"
        "TOOL_RESULT: [{\"name\":\"m.py\",\"type\":\"file\"}]"
    )
    assert _filesystem_lexical_tool(prompt, _tools()) == "write_file"
    assert _text_arg(prompt, "content") == ["1"]


def test_filesystem_output_path_ignores_filenames_in_tool_result() -> None:
    prompt = (
        "Write the answer to structure_analysis.txt in the main directory. Main directory: "
        "/private/tmp/mcpmark-fs-state/extracted/folder_structure\n"
        "ASSISTANT: <tool_call>{\"arguments\":{\"path\":\"/tool_call\"}}</tool_call>\n"
        "TOOL_RESULT: {\"text\": \"[{\\\"name\\\":\\\"report_2.txt\\\"}]\"}"
    )
    assert _workspace_output_path(prompt) == [
        "/private/tmp/mcpmark-fs-state/extracted/folder_structure/structure_analysis.txt"
    ]


def test_count_parser_accepts_backtick_extension_and_structured_tree() -> None:
    prompt = (
        "Count the total number of `.py` files in all subdirectories. Write the answer to "
        "structure_analysis.txt. Main directory: /tmp/workspace\n"
        "TOOL_RESULT: [{\"name\":\"a.py\",\"type\":\"file\"}, "
        "{\"name\":\"b.py\",\"type\":\"file\"}]"
    )
    assert _text_arg(prompt, "content") == ["2"]

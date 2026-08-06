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


def test_create_directory_then_empty_file_switches_to_write() -> None:
    prompt = (
        "Create a folder named `final_version`, then create an empty file named "
        "`agreement_v10.txt` inside it. Workspace root: /tmp/legal_files\n"
        "ASSISTANT: <tool_call>{\"name\":\"create_directory\",\"arguments\":{}}"
        "</tool_call>\nTOOL_RESULT: directory created"
    )
    assert _filesystem_lexical_tool(prompt, _tools()) == "write_file"
    assert _text_arg(prompt, "content") == [""]


def test_output_path_prefers_explicit_artifact_over_source_files() -> None:
    prompt = (
        "Read file_01.txt through file_20.txt and create a file named `answer.txt`. "
        "Workspace root: /tmp/file_context\nTOOL_RESULT: done"
    )
    assert _workspace_output_path(prompt) == ["/tmp/file_context/answer.txt"]


def test_output_path_preserves_parent_and_target_directories() -> None:
    prompt = (
        'Create a folder named `final_version` inside the folder "legal_files/" directory. '
        "Create an empty file with the same name as Preferred_Stock_Purchase_Agreement_v10.txt. "
        "Workspace root: /tmp/legal_document\nTOOL_RESULT: directory created"
    )
    assert _workspace_output_path(prompt) == [
        "/tmp/legal_document/legal_files/final_version/Preferred_Stock_Purchase_Agreement_v10.txt"
    ]

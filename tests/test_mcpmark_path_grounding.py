"""Regression tests for generic filesystem path grounding."""

from localagent.agent.constrained import _arg_options, _path


def test_path_prefers_explicit_absolute_workspace_over_task_identifier() -> None:
    prompt = (
        "MCPMark filesystem task file_context/file_splitting. Create the output directory. "
        "Workspace root: /private/tmp/mcpmark-fs-transfer/workspace"
    )
    assert _path(prompt) == ["/private/tmp/mcpmark-fs-transfer/workspace"]


def test_unformatted_path_schema_uses_path_extractor() -> None:
    prompt = "Create the directory at /private/tmp/mcpmark-fs-transfer/workspace/split."
    schema = {"type": "string"}
    assert _arg_options(prompt, "path", schema, True) == [
        "/private/tmp/mcpmark-fs-transfer/workspace/split"
    ]


def test_named_directory_is_joined_to_workspace_root() -> None:
    prompt = (
        "Create a new directory named `split` in the test directory. "
        "Workspace root: /private/tmp/mcpmark-fs-transfer/workspace"
    )
    assert _path(prompt) == ["/private/tmp/mcpmark-fs-transfer/workspace/split"]

"""Standard agent tool schemas (JSON-schema parameters) used by eval + the grounded decoder.

These carry real `parameters` so the grounded decoder can be **schema-driven** (read arg names,
types, enums, required) instead of hardcoding per-tool extraction. Mirrors configs/data/tool_pool.json.
"""

from __future__ import annotations

from localagent.data.schema import ToolSpec

STANDARD_TOOLS = [
    ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "format": "arithmetic"}},
            "required": ["expression"],
        },
    ),
    ToolSpec(
        name="web_search",
        description="Search the web.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="planner",
        description="Make a plan to achieve a goal.",
        parameters={
            "type": "object",
            "properties": {"goal": {"type": "string"}},
            "required": ["goal"],
        },
    ),
    ToolSpec(
        name="define",
        description="Define a term.",
        parameters={"type": "object", "properties": {"term": {"type": "string"}},
                    "required": ["term"]},
    ),
    ToolSpec(
        name="play_music",
        description="Play a song.",
        parameters={"type": "object", "properties": {"song": {"type": "string"}},
                    "required": ["song"]},
    ),
    ToolSpec(
        name="get_news",
        description="Get news on a topic.",
        parameters={"type": "object", "properties": {"topic": {"type": "string"}},
                    "required": ["topic"]},
    ),
    # --- coding-agent tools (Claude Code / Codex-style) ---
    ToolSpec(
        name="read_file", description="Read a file.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="write_file", description="Create or write a file.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="grep_search", description="Search the codebase for a pattern.",
        parameters={"type": "object",
                    "properties": {"pattern": {"type": "string", "format": "quoted"}},
                    "required": ["pattern"]},
    ),
    ToolSpec(
        name="run_command", description="Run a shell command.",
        parameters={"type": "object",
                    "properties": {"command": {"type": "string", "format": "quoted"}},
                    "required": ["command"]},
    ),
    ToolSpec(
        name="git_commit", description="Make a git commit.",
        parameters={"type": "object",
                    "properties": {"message": {"type": "string", "format": "quoted"}},
                    "required": ["message"]},
    ),
    ToolSpec(
        name="run_tests", description="Run the test suite.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    # --- popular everyday tools ---
    ToolSpec(
        name="set_reminder", description="Set a reminder for a task.",
        parameters={"type": "object", "properties": {"task": {"type": "string"}},
                    "required": ["task"]},
    ),
    ToolSpec(
        name="set_timer", description="Set a timer for a duration.",
        parameters={"type": "object", "properties": {"duration": {"type": "string"}},
                    "required": ["duration"]},
    ),
]

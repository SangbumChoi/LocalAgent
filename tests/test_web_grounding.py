from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
WEB_APP = ROOT / "spaces" / "localagent-webgpu" / "app.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_typed_schema_grounding_matches_action_contracts():
    cases = [
        {
            "prompt": "Drag 'the notification bell' onto 'the Done button'.",
            "schema": {
                "properties": {
                    "source": {"type": "string", "format": "quoted"},
                    "dest": {"type": "string", "format": "quoted"},
                },
                "required": ["source", "dest"],
            },
            "expected": {"source": "the notification bell", "dest": "the Done button"},
        },
        {
            "prompt": "Send an Escape keypress.",
            "schema": {
                "properties": {
                    "key": {"type": "string", "enum": ["Enter", "Tab", "Escape"]},
                },
                "required": ["key"],
            },
            "expected": {"key": "Escape"},
        },
        {
            "prompt": "Give it 12 seconds.",
            "schema": {
                "properties": {"seconds": {"type": "integer"}},
                "required": ["seconds"],
            },
            "expected": {"seconds": 12},
        },
        {
            "prompt": "Navigate to huggingface.co in the browser.",
            "schema": {
                "properties": {"url": {"type": "string", "format": "url"}},
                "required": ["url"],
            },
            "expected": {"url": "huggingface.co"},
        },
        {
            "prompt": "Show me the contents of api/routes.go.",
            "schema": {
                "properties": {"path": {"type": "string", "format": "path"}},
                "required": ["path"],
            },
            "expected": {"path": "api/routes.go"},
        },
        {
            "prompt": "Turn caching off.",
            "schema": {
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
            "expected": {"enabled": False},
        },
    ]
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
const cases = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(
  cases.map((item) => groundFromSchema(item.prompt, item.schema))
));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP), json.dumps(cases)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [case["expected"] for case in cases]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_grounding_rejects_missing_required_value():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
const schema = {
  properties: { direction: { type: "string", enum: ["up", "down"] } },
  required: ["direction"],
};
process.stdout.write(JSON.stringify(groundFromSchema("Please move somehow.", schema)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) is None

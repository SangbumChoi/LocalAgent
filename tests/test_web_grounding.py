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


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_stateful_grounding_ignores_goal_and_observation_quotes():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema, compactDispatchQuery } = require(process.argv[1]);
const prompt = 'Goal: compose and fill an email. Current state JSON: {"app":"home"}. ' +
  'Next required action: Open the Gmail app on the Android phone. Return exactly one structured action';
const app = groundFromSchema(prompt, {
  properties: {app_name: {type: "string", format: "quoted"}}, required: ["app_name"]
});
const urlPrompt = 'Goal: search mail. Current state JSON: {"page":null}. ' +
  'Next required action: Open https://example.local/mail in the browser.';
const url = groundFromSchema(urlPrompt, {
  properties: {url: {type: "string", format: "url"}}, required: ["url"]
});
const compact = compactDispatchQuery(urlPrompt);
process.stdout.write(JSON.stringify({app, url, compact}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "app": {"app_name": "Gmail"},
        "url": {"url": "https://example.local/mail"},
        "compact": "Open https://example.local/mail in the browser.",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_url_grounding_strips_serialized_user_marker_from_pointer_span():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
const schema = { properties: { url: {type: "string", format: "url"} }, required: ["url"] };
process.stdout.write(JSON.stringify(groundFromSchema(
  "Open https://example.com",
  schema,
  { url: "<|user|>Open https://example.com" }
)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"url": "https://example.com"}

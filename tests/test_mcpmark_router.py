from __future__ import annotations

import json

import pytest

from localagent.eval.mcpmark_router import (
    SERVICE_ROUTES,
    SERVICE_TOOL_FAMILIES,
    _discover,
)


def _write_task(root, service="notion", suite="standard"):
    task = root / "tasks" / service / suite / "category" / "task"
    task.mkdir(parents=True)
    (task / "meta.json").write_text(
        json.dumps({"category_name": "Category", "difficulty": "L1"}), encoding="utf-8"
    )
    (task / "description.md").write_text("Do the task.", encoding="utf-8")


def test_discover_mcpmark_router_tasks_and_service_contract(tmp_path):
    _write_task(tmp_path)
    rows = _discover(tmp_path, "standard")
    assert len(rows) == 1
    assert rows[0][0] == "notion"
    assert "notion_create_page" in SERVICE_TOOL_FAMILIES["notion"]
    assert SERVICE_ROUTES["notion"] == frozenset({"app_action"})


def test_discover_mcpmark_router_rejects_missing_description(tmp_path):
    task = tmp_path / "tasks" / "postgres" / "standard" / "category" / "task"
    task.mkdir(parents=True)
    (task / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing description"):
        _discover(tmp_path, "standard")

from pathlib import Path

import yaml


SUPPLEMENTAL = Path(__file__).parents[1] / "configs/data/realistic-agent-eval.supplemental.yaml"


def test_supplemental_realistic_sources_are_explicitly_catalog_only() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_realistic_agent_supplemental_catalog"
    entries = payload["entries"]
    assert len(entries) == 8
    assert {entry["id"] for entry in entries} == {
        "androidworld",
        "browsergym_miniwob",
        "computer_agent_arena",
        "cua_lite_agentnet",
        "enterpriseopsgym",
        "mcpmark",
        "osworld2_trajectory",
        "toolsandbox",
    }
    for entry in entries:
        assert entry["source_url"].startswith(("https://github.com/", "https://huggingface.co/"))
        assert entry["split_policy"]
        assert entry["runtime"]
        assert entry["webgpu_projection"]


def test_mcpmark_revision_matches_the_published_metadata_profile() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    entry = next(row for row in payload["entries"] if row["id"] == "mcpmark")
    assert entry["source_revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"

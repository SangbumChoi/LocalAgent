from pathlib import Path

import yaml


SUPPLEMENTAL = Path(__file__).parents[1] / "configs/data/realistic-agent-eval.supplemental.yaml"


def test_supplemental_realistic_sources_are_explicitly_catalog_only() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_realistic_agent_supplemental_catalog"
    entries = payload["entries"]
    assert len(entries) == 14
    assert {entry["id"] for entry in entries} == {
        "androidworld",
        "browsergym_miniwob",
        "iosworld",
        "mobile_safety_bench",
        "androidcontrol_curated",
        "osworld_mcp",
        "computer_agent_arena",
        "cua_lite_agentnet",
        "enterpriseopsgym",
        "mcpmark",
        "osworld2_trajectory",
        "toolsandbox",
        "webbench",
        "bu_bench_v1",
    }
    for entry in entries:
        assert entry["source_url"].startswith(
            ("https://github.com/", "https://huggingface.co/", "https://mobilesafetybench.github.io/")
        )
        assert entry["split_policy"]
        assert entry["runtime"]
        assert entry["webgpu_projection"]


def test_mcpmark_revision_matches_the_published_metadata_profile() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    entry = next(row for row in payload["entries"] if row["id"] == "mcpmark")
    assert entry["source_revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"


def test_toolsandbox_revision_matches_the_published_metadata_profile() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    entry = next(row for row in payload["entries"] if row["id"] == "toolsandbox")
    assert entry["source_revision"] == "165848b9a78cead7ca7fe7c89c688b58e6501219"
    assert entry["scale"].startswith("129_base_scenarios")


def test_live_browser_sources_are_eval_only_and_contamination_safe() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in payload["entries"]}
    for source_id in ("webbench", "bu_bench_v1"):
        entry = entries[source_id]
        assert entry["split_policy"].startswith("evaluation_only")
        assert "training" in entry["split_policy"]
        assert entry["source_revision"] == "pin-before-use"
    assert "encrypted" in entries["bu_bench_v1"]["split_policy"]

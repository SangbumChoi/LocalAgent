from pathlib import Path

import yaml


SUPPLEMENTAL = Path(__file__).parents[1] / "configs/data/realistic-agent-eval.supplemental.yaml"


def test_supplemental_realistic_sources_are_explicitly_catalog_only() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_realistic_agent_supplemental_catalog"
    entries = payload["entries"]
    assert len(entries) == 21
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
        "cua_gym",
        "osworld_verified_trajectories",
        "androidlab",
        "knowu_bench",
        "appagent_benchmark",
        "groundcua",
        "ui_tars_action_contract",
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


def test_cua_gym_and_osworld_trajectory_sources_are_not_silently_trainable() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in payload["entries"]}
    cua = entries["cua_gym"]
    assert cua["license"] == "CC-BY-4.0"
    assert cua["source_revision"] == "3c021d0"
    assert "no official evaluation partition" in cua["notes"]
    osworld = entries["osworld_verified_trajectories"]
    assert osworld["license"] == "MIT"
    assert osworld["source_revision"] == "8413d635f654c1f95a17f8813f52f2b1b450c566"
    assert osworld["split_policy"].startswith("evaluation_only")


def test_new_mobile_and_grounding_sources_fail_closed_until_released() -> None:
    payload = yaml.safe_load(SUPPLEMENTAL.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in payload["entries"]}
    assert entries["knowu_bench"]["source_revision"] == "c03a825991ede13add6631f2ed19b90755930dc6"
    assert "hidden_profiles" in entries["knowu_bench"]["split_policy"]
    assert entries["appagent_benchmark"]["source_revision"] == "2c1900422caf6f9e94e96d5dd984b530e5a5fbf8"
    assert entries["groundcua"]["source_revision"] == "94cba61693d6258c7100f7016a299b84b1bb7732"
    assert "dataset_terms" in entries["groundcua"]["split_policy"]
    assert "screenshots" in entries["groundcua"]["notes"]
    assert entries["ui_tars_action_contract"]["source_revision"] == "582f3a7ea5d285ee8ed9e2e84048d1ab01453c49"

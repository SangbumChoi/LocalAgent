"""Regression checks for the source-bound cross-surface continuation receipt."""

import json
from pathlib import Path


def test_cross_surface_receipt_binds_public_sources_and_mixed_transfer() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    receipt = json.loads((root / "m100-cross-surface-public-continuation-v1.json").read_text())
    assert receipt["kind"] == "localagent_cross_surface_public_continuation_report"
    assert receipt["rows"] == {"train": 4629, "eval": 1041}
    assert [source["label"] for source in receipt["train_sources"]] == [
        "mobile",
        "desktop",
        "browser",
    ]
    assert [source["label"] for source in receipt["eval_sources"]] == [
        "mobile",
        "desktop",
        "browser",
    ]
    references = {
        source["label"]: source["public_reference"] for source in receipt["train_sources"]
    }
    assert references["mobile"]["dataset"] == "OfficerChul/Android-Control-84k"
    assert references["desktop"]["dataset"] == "xlangai/AgentNet"
    assert references["browser"]["dataset"] == "osunlp/Mind2Web"
    assert receipt["before"]["eval"]["assistant_token_accuracy"] < receipt["after"]["eval"]["assistant_token_accuracy"]
    assert receipt["before"]["eval_by_source"]["mobile"]["assistant_token_accuracy"] < receipt["after"]["eval_by_source"]["mobile"]["assistant_token_accuracy"]
    assert receipt["before"]["eval_by_source"]["desktop"]["assistant_token_accuracy"] > receipt["after"]["eval_by_source"]["desktop"]["assistant_token_accuracy"]
    groups = receipt["weight_transfer"]["groups"]
    assert groups["action_heads"]["relative_delta_l2"] == 0.0
    assert 0.004 < groups["embedding"]["relative_delta_l2"] < 0.006
    assert "no official benchmark score" in receipt["claim_boundary"]

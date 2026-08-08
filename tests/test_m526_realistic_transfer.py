import json
from pathlib import Path


def test_m526_public_transfer_receipt_is_current_and_fail_closed() -> None:
    raw = Path("docs/paper/results/raw")
    receipt = json.loads(
        (raw / "m526-realistic-cross-surface-warm-random-8step-combined-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["parent_checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert {source["label"] for source in receipt["sources"]} == {
        "androidcontrol",
        "agentnet",
        "mind2web",
        "mcpmark",
    }
    assert receipt["comparison"]["aggregate"]["warm_start_better_after"] is True
    assert receipt["comparison"]["aggregate"]["random_after_token_accuracy"] == 0.0
    assert receipt["arms"]["warm"]["after"]["assistant_sequence_accuracy"] == 0.0
    assert "native emulator/browser/desktop/MCP execution" in receipt["interpretation"]["not_claimed"]


def test_m528_gate_keeps_publication_blocked() -> None:
    gate = json.loads(
        Path("docs/paper/results/raw/m528-workshop-gate-current-m526-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["ready"] is False
    weight = next(
        check for check in gate["checks"] if check["requirement"] == "weights:transfer_and_no_transfer_ablation"
    )
    assert weight["status"] == "pass"
    assert any(item["requirement"] == "artifacts:public_model_demo_manifest" for item in gate["blocking_requirements"])

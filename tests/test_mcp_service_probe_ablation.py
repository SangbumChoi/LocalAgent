from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.train_mcp_service_probe_ablation import MCPMARK_REVISION, _receipt_sha256


def test_ablation_binds_published_transfer_receipt() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m38-mcp-service-contract-probe-v1.json"
    assert MCPMARK_REVISION == json.loads(path.read_text(encoding="utf-8"))["dataset"]["revision"]
    assert len(_receipt_sha256(path)) == 64


def test_published_no_transfer_receipt_is_self_hashed_and_matched() -> None:
    path = Path(__file__).parents[1] / (
        "docs/paper/results/raw/m53-mcp-service-contract-no-transfer-v1.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["comparison"]["rows"] == 239
    assert receipt["comparison"]["random_minus_transfer"]["selector_top1"] < -0.1

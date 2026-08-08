"""Integrity checks for the m586 native BrowserGym receipt."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m586-browsergym-m585-warm-full-v1.json"


def test_m586_receipt_is_checkpoint_bound_and_official_plan() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["source"]["official_split_verified"] is True
    assert data["checkpoint"]["child_sha256"] == (
        "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"
    )
    result = data["result"]
    assert result["task_count"] == 240
    assert result["passed_tasks"] == 5
    assert result["success_rate"] == 5 / 240
    assert result["grounded_steps"] == 219
    assert result["noop_or_ungrounded_steps"] == 1960
    assert result["action_errors"] == 0
    assert result["vision_used"] is False
    assert result["coordinate_fallback"] is False
    assert result["semantic_fallback"] is False


def test_m586_receipt_self_hash() -> None:
    data = json.loads(RECEIPT.read_text())
    declared = data.pop("receipt_self_sha256")
    assert declared
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared

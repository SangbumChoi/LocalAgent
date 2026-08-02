import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m125_mobilegym_answer_adaptation_is_hash_bound_and_non_gating() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m125-mobilegym-answer-adaptation-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["native_benchmark_id"] == "mobilegym"
    assert receipt["benchmark_id"] == "mobilegym_model_adaptation_probe"
    assert receipt["native_receipt_eligible"] is False
    assert receipt["environment_executed"] is True
    assert receipt["official_split"] == "test"
    assert receipt["official_split_verified"] is True
    assert receipt["data"]["train_rows"] == 4096
    assert receipt["data"]["eval_rows"] == 904
    assert receipt["training"]["tool_pool_size"] == 63
    arms = receipt["training"]["arms"]
    assert set(arms) == {"warm", "random"}
    for arm in arms.values():
        assert arm["native_success_rate"] == 0.0
        assert arm["native_judge_passed"] is False
        assert arm["native_judge_issue_count"] == 3
        assert arm["weight_report"]["shared_tensors"] == 51
        assert arm["weight_report"]["config_compatible"] is True
        assert arm["weight_report"]["tokenizer_compatible"] is True

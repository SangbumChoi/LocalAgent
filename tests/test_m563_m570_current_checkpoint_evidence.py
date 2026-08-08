import hashlib
import json
from pathlib import Path


ROOT = Path("docs/paper/results/raw")
CURRENT_SHA = "43b32b15899bcd97d7a822ef8441b04521d863b30c23c17080dbfaf4f0a14d7c"


def _load_verified(name: str) -> dict:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    return payload


def test_m563_lineage_chain_is_checkpoint_bound() -> None:
    payload = _load_verified("m563-m553-lineage-training-v1.json")
    assert payload["chain"]["grounded_child"]
    assert payload["chain"]["multisurface_child"] == payload["candidate_checkpoint"]["sha256"]
    assert payload["candidate_checkpoint"]["lineage_stage"] == "sft"


def test_m564_rl_preflight_has_gate_visible_lineage() -> None:
    payload = _load_verified("m564-m553-stateful-rl-preflight-v1.json")
    assert payload["status"] == "passed"
    assert payload["metrics"]["lineage"]["parent_checkpoint_sha256"] == CURRENT_SHA
    assert payload["measurement"]["realized_optimizer_updates"] == 2


def test_m565_transfer_ablation_has_matched_random_control() -> None:
    payload = _load_verified("m565-m553-transfer-ablation-v1.json")
    assert payload["parent_checkpoint"]["sha256"] == CURRENT_SHA
    comparison = payload["comparison"]
    assert comparison["aggregate"]["warm_start_better_after"] is True
    assert comparison["surfaces"].keys() == {"androidcontrol", "agentnet"}
    assert payload["weight_transfer_analysis"]["warm"]["compatibility"]["config_mismatches"] == {}


def test_m566_webgpu_capability_is_native_and_checkpoint_bound() -> None:
    payload = _load_verified("m566-m553-webgpu-capability-v1.json")
    assert payload["checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["backend"] == "webgpu"
    assert payload["environment_executed"] is True
    assert payload["capability"]["exact_actions"] == 3
    assert payload["performance"]["tokens_per_second_p50"] >= 100


def test_m567_m568_native_receipts_are_official_split_negative_controls() -> None:
    mobile = _load_verified("m567-m553-mobilegym-native-full-v1.json")
    browser = _load_verified("m568-m553-browsergym-native-full-v1.json")
    assert mobile["checkpoint_sha256"] == CURRENT_SHA
    assert browser["checkpoint_sha256"] == CURRENT_SHA
    assert mobile["environment"]["official_split_verified"] is True
    assert browser["environment"]["official_split_verified"] is True
    assert mobile["result"]["success_rate"] == 1 / 256
    assert browser["result"]["success_rate"] == 5 / 240


def test_m570_workshop_gate_is_fail_closed_with_passed_current_checks() -> None:
    payload = _load_verified("m570-workshop-gate-current-m553-v1.json")
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["checks"]["native_mobilegym"] == "pass"
    assert payload["checks"]["native_browsergym_miniwob"] == "pass"
    assert payload["checks"]["webgpu_native_capability_and_latency"] == "pass"
    assert payload["checks"]["weights_transfer_and_no_transfer_ablation"] == "pass"
    assert payload["checks"]["training_rl_preflight"] == "pass"
    assert "native:androidworld" in payload["blocking_requirements"]
    assert "artifacts:public_model_demo_manifest" in payload["blocking_requirements"]

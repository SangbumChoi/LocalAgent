"""Integrity checks for the current stateful-productivity RL and weight audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RL_RECEIPT = ROOT / "docs/paper/results/raw/m234-stateful-productivity-grpo-head-preserved-v1.json"
WEIGHT_RECEIPT = ROOT / "docs/paper/results/raw/m235-stateful-productivity-rl-weight-transfer-v1.json"
DEPLOY_RECEIPT = ROOT / "docs/paper/results/raw/m236-stateful-productivity-rl-deployment-smoke-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m234_rl_receipt_is_self_hashed_and_preserves_deployment_heads() -> None:
    payload = json.loads(RL_RECEIPT.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    assert _canonical_hash(payload) == expected
    assert payload["kind"] == "localagent_stateful_productivity_grpo_simulation"
    assert payload["source"]["native_runtime_executed"] is False
    assert payload["source"]["public_benchmark_text_used"] is False
    assert payload["configuration"]["deployment_heads_trainable"] is False
    assert set(payload["configuration"]["deployment_heads_preserved"]) == {
        "tool_head",
        "ptr_head",
        "route_head",
        "dense_selector",
        "selector_proj",
    }
    assert payload["training"]["exact_match_accuracy_post"] == 0.0
    assert payload["training"]["mean_reward_post"] > payload["training"]["mean_reward_pre"]


def test_m235_weight_audit_is_compatible_and_freezes_action_heads() -> None:
    payload = json.loads(WEIGHT_RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_weight_transfer_analysis"
    compatibility = payload["compatibility"]
    assert compatibility["config_mismatches"] == {}
    assert compatibility["shape_mismatches"] == {}
    assert compatibility["removed_tensors"] == []
    assert compatibility["tokenizer_sha256_equal"] is True
    assert payload["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert payload["groups"]["embedding"]["relative_delta_l2"] > 0.0
    assert payload["groups"]["ffn"]["relative_delta_l2"] > 0.0


def test_m236_rl_child_is_loadable_but_not_deployment_approved() -> None:
    payload = json.loads(DEPLOY_RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_local_deployment_smoke"
    assert payload["environment"]["environment_executed"] is False
    assert payload["environment"]["external_accounts"] is False
    assert payload["summary"]["cases"] == 10
    assert payload["summary"]["exact_tool"] == 1
    assert payload["summary"]["tool_accuracy"] == 0.1

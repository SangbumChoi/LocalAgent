import json
from pathlib import Path

from localagent.eval.workshop_gate import build_workshop_gate, write_workshop_gate


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "configs/data/realistic-agent-eval.catalog.yaml"


def test_default_gate_is_not_ready_and_lists_native_blockers() -> None:
    report = build_workshop_gate(CATALOG, repo_root=ROOT)
    assert report["ready"] is False
    requirements = {item["requirement"] for item in report["blocking_requirements"]}
    assert "native:androidworld" in requirements
    assert "native:agentnet" in requirements
    assert "webgpu:native_capability_and_latency" in requirements
    assert "artifacts:public_model_demo_manifest" in requirements


def test_native_receipt_contract_requires_execution_and_split(tmp_path: Path) -> None:
    receipt = tmp_path / "androidworld.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_id": "androidworld",
                "environment_executed": True,
                "official_split_verified": True,
                "task_count": 1,
                "success_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"androidworld": receipt},
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:androidworld")
    assert check["status"] == "pass"


def test_native_browsergym_probe_is_recorded_but_not_official_score() -> None:
    receipt = ROOT / "docs/paper/results/raw/m29-browsergym-native-model-eval-v1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["environment_executed"] is True
    assert payload["task_count"] == 240
    assert payload["success_rate"] == 0.0
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"browsergym_miniwob": receipt},
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:browsergym_miniwob")
    assert check["status"] == "blocked"
    assert check["blockers"] == ["official_split_not_verified"]


def test_weight_gate_requires_two_compatible_labeled_reports(tmp_path: Path) -> None:
    payload = {
        "compatibility": {
            "config_mismatches": {},
            "shape_mismatches": {},
            "tokenizer_sha256_equal": True,
        },
        "ablation": "parent_heads",
        "held_out": {"selector": 0.5},
    }
    first = tmp_path / "parent.json"
    second = tmp_path / "random.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    payload["ablation"] = "random"
    second.write_text(json.dumps(payload), encoding="utf-8")
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        weight_reports=[first, second],
    )
    check = next(item for item in report["checks"] if item["requirement"] == "weights:transfer_and_no_transfer_ablation")
    assert check["status"] == "pass"


def test_weight_gate_accepts_combined_parent_vs_random_receipt() -> None:
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        weight_reports=["docs/paper/results/raw/m25-weight-transfer-ablation-v1.json"],
    )
    check = next(item for item in report["checks"] if item["requirement"] == "weights:transfer_and_no_transfer_ablation")
    assert check["status"] == "pass"


def test_write_gate_refuses_overwrite(tmp_path: Path) -> None:
    report = build_workshop_gate(CATALOG, repo_root=ROOT)
    output = tmp_path / "gate.json"
    write_workshop_gate(report, output)
    assert json.loads(output.read_text(encoding="utf-8"))["ready"] is False
    try:
        write_workshop_gate(report, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("write_workshop_gate must refuse overwrite")

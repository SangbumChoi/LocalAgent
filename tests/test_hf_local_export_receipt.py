import json
from pathlib import Path


RECEIPT = Path(__file__).parents[1] / "docs/paper/results/raw/m26-hf-local-export-v2.json"


def test_hf_local_export_receipt_is_not_a_publication_claim() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_hf_local_export_receipt"
    assert payload["publication"] == {
        "published": False,
        "hub_url": None,
        "reason": "HF authentication was not configured; this receipt binds a local bundle only.",
    }
    assert payload["bundle"]["parameters"] == 10524544
    assert payload["bundle"]["tokenizer"]["kind"] == "bpe"
    assert set(payload["bundle"]["head_keys"]) == {
        "dense_selector",
        "ptr_head",
        "route_head",
        "selector_proj",
        "tool_head",
    }


def test_m131_mobile_export_binds_dispatch_metadata_without_claiming_upload() -> None:
    payload = json.loads(
        Path("docs/paper/results/raw/m131-hf-local-export-m129-mobile-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["kind"] == "localagent_hf_local_export_receipt"
    assert payload["parameter_count"] == 10524544
    assert payload["dispatch_tool_count"] == 63
    assert payload["tokenizer_sha256"] == "8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c"
    assert payload["export_verified_locally"] is True
    assert payload["hub"] == {
        "authenticated": False,
        "uploaded": False,
        "reason": "hf auth whoami reports no login; upload requires a user-provided Hugging Face token and repository",
    }
    assert payload["bundle"]["agent_heads.bin"]["bytes"] == 10137222

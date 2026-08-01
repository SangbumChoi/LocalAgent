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

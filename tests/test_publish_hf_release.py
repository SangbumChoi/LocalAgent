import hashlib
import json
from pathlib import Path

import pytest

from scripts.publish_hf_release import _copy_static_space, _token


def test_publish_token_prefers_cli_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    assert _token(None) == "environment-token"
    assert _token("cli-token") == "cli-token"


def test_space_staging_excludes_generated_bundle_and_maintainer_docs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "index.html").write_text("app", encoding="utf-8")
    (source / "DEPLOY.md").write_text("maintainer-only", encoding="utf-8")
    (source / "model.onnx").write_bytes(b"stale")
    (source / "bundle-manifest.json").write_text("{}", encoding="utf-8")
    _copy_static_space(source, target)
    assert (target / "index.html").read_text(encoding="utf-8") == "app"
    assert not (target / "DEPLOY.md").exists()
    assert not (target / "model.onnx").exists()
    assert not (target / "bundle-manifest.json").exists()


def test_m331_local_receipt_binds_current_checkpoint_without_public_claim() -> None:
    receipt = Path("docs/paper/results/raw/m331-hf-paired-release-local-v1.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["webgpu"]["parity_gate"] is True
    assert payload["space"]["verified"] is True
    assert payload["publication"] == {
        "authenticated": False,
        "published": False,
        "reason": "local-only preparation; HF authentication was not configured",
        "uploaded": False,
    }

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from localagent.model import LocalAgentLM, ModelConfig
from localagent.train import stage_data
from localagent.train.stage_data import (
    LINEAGE_VERSION,
    build_stage_lineage,
    load_stage_parent_checkpoint,
    tokenizer_identity,
)


def _config(*, rope_theta: float = 10_000.0) -> ModelConfig:
    cfg = ModelConfig(
        name="stage-parent-lineage-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=64,
        rope_theta=rope_theta,
        dropout=0.0,
    )
    cfg.assert_within_budget()
    return cfg


def _tokenizer_sha256() -> str:
    return str(tokenizer_identity("byte", vocab_size=256)["sha256"])


def _checkpoint(
    cfg: ModelConfig,
    *,
    stage: str,
    lineage_stage: str | None = None,
    version: object = LINEAGE_VERSION,
    lineage_tokenizer_sha256: object | None = None,
    tokenizer_sha256: object | None = None,
) -> dict:
    expected_tokenizer = _tokenizer_sha256()
    lineage: dict[str, object] = {
        "version": version,
        "stage": lineage_stage or stage,
    }
    if lineage_tokenizer_sha256 is not None:
        lineage["tokenizer_sha256"] = lineage_tokenizer_sha256
    payload = {
        "cfg": cfg.__dict__,
        "state_dict": LocalAgentLM(cfg).state_dict(),
        "stage": stage,
        "lineage": lineage,
    }
    if tokenizer_sha256 is not None:
        payload["tokenizer"] = {
            "kind": "byte",
            "sha256": tokenizer_sha256,
        }
    if lineage_tokenizer_sha256 is None and tokenizer_sha256 is None:
        return payload
    lineage.setdefault("tokenizer_sha256", expected_tokenizer)
    return payload


@pytest.mark.parametrize(
    ("stage", "parent_stage"),
    [
        ("midtrain", "pretrain"),
        ("sft", "midtrain"),
        ("rl", "sft"),
    ],
)
def test_parent_loader_requires_the_exact_stage_chain_and_returns_loaded_byte_sha(
    tmp_path: Path,
    stage: str,
    parent_stage: str,
) -> None:
    cfg = _config()
    tokenizer_sha256 = _tokenizer_sha256()
    path = tmp_path / f"{parent_stage}.pt"
    torch.save(
        _checkpoint(
            cfg,
            stage=parent_stage,
            lineage_tokenizer_sha256=tokenizer_sha256,
            tokenizer_sha256=tokenizer_sha256,
        ),
        path,
    )
    expected_parent_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    checkpoint, parent_sha256 = load_stage_parent_checkpoint(
        path,
        stage=stage,
        requested_model_config=cfg,
        expected_tokenizer_sha256=tokenizer_sha256,
    )

    assert checkpoint["stage"] == parent_stage
    assert parent_sha256 == expected_parent_sha256


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"stage": "sft"}, "requires an exact pretrain parent"),
        ({"lineage_stage": "midtrain"}, "requires an exact pretrain parent"),
        ({"version": LINEAGE_VERSION + 1}, "unsupported parent checkpoint lineage version"),
        ({"version": True}, "unsupported parent checkpoint lineage version"),
    ],
)
def test_parent_loader_rejects_wrong_stage_or_lineage_version(
    tmp_path: Path,
    mutation: dict,
    error: str,
) -> None:
    cfg = _config()
    tokenizer_sha256 = _tokenizer_sha256()
    path = tmp_path / "parent.pt"
    parent_stage = str(mutation.get("stage", "pretrain"))
    checkpoint_mutation = {
        key: value for key, value in mutation.items() if key != "stage"
    }
    torch.save(
        _checkpoint(
            cfg,
            stage=parent_stage,
            lineage_tokenizer_sha256=tokenizer_sha256,
            tokenizer_sha256=tokenizer_sha256,
            **checkpoint_mutation,
        ),
        path,
    )

    with pytest.raises(ValueError, match=error):
        load_stage_parent_checkpoint(
            path,
            stage="midtrain",
            requested_model_config=cfg,
            expected_tokenizer_sha256=tokenizer_sha256,
        )


@pytest.mark.parametrize(
    ("lineage_tokenizer", "tokenizer", "error"),
    [
        (None, None, "no content-bound tokenizer identity"),
        ("", None, "lineage.tokenizer_sha256 sha256"),
        ("a" * 64, "b" * 64, "conflicting tokenizer identities"),
        ("b" * 64, "b" * 64, "does not match configured tokenizer"),
    ],
)
def test_parent_loader_rejects_missing_empty_conflicting_or_wrong_tokenizer_identities(
    tmp_path: Path,
    lineage_tokenizer: object | None,
    tokenizer: object | None,
    error: str,
) -> None:
    cfg = _config()
    path = tmp_path / "parent.pt"
    torch.save(
        _checkpoint(
            cfg,
            stage="pretrain",
            lineage_tokenizer_sha256=lineage_tokenizer,
            tokenizer_sha256=tokenizer,
        ),
        path,
    )

    with pytest.raises(ValueError, match=error):
        load_stage_parent_checkpoint(
            path,
            stage="midtrain",
            requested_model_config=cfg,
            expected_tokenizer_sha256=_tokenizer_sha256(),
        )


def test_parent_loader_rejects_architecture_mismatch(tmp_path: Path) -> None:
    configured = _config()
    parent_cfg = _config(rope_theta=20_000.0)
    tokenizer_sha256 = _tokenizer_sha256()
    path = tmp_path / "parent.pt"
    torch.save(
        _checkpoint(
            parent_cfg,
            stage="pretrain",
            lineage_tokenizer_sha256=tokenizer_sha256,
            tokenizer_sha256=tokenizer_sha256,
        ),
        path,
    )

    with pytest.raises(ValueError, match="rope_theta"):
        load_stage_parent_checkpoint(
            path,
            stage="midtrain",
            requested_model_config=configured,
            expected_tokenizer_sha256=tokenizer_sha256,
        )


def test_parent_lineage_uses_loaded_bytes_after_path_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    tokenizer_sha256 = _tokenizer_sha256()
    path = tmp_path / "parent.pt"
    torch.save(
        _checkpoint(
            cfg,
            stage="pretrain",
            lineage_tokenizer_sha256=tokenizer_sha256,
            tokenizer_sha256=tokenizer_sha256,
        ),
        path,
    )
    original_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(candidate: Path) -> bytes:
        nonlocal reads
        if candidate == path:
            reads += 1
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    _, loaded_sha256 = load_stage_parent_checkpoint(
        path,
        stage="midtrain",
        requested_model_config=cfg,
        expected_tokenizer_sha256=tokenizer_sha256,
    )
    assert reads == 1
    torch.save({"mutated": True}, path)
    monkeypatch.setattr(
        stage_data,
        "sha256_file",
        lambda _path: pytest.fail("build_stage_lineage reopened the mutable parent path"),
    )

    lineage = build_stage_lineage(
        stage="midtrain",
        config={"stage": "midtrain"},
        model_config=cfg.__dict__,
        data_identity={"source": "test"},
        tokenizer={"sha256": tokenizer_sha256},
        workspace=tmp_path,
        parent_checkpoint_sha256=loaded_sha256,
    )

    assert loaded_sha256 == original_sha256
    assert lineage["parent_checkpoint_sha256"] == original_sha256

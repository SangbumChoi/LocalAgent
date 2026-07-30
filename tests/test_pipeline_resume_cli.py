"""Operational exact-resume CLI contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from localagent import cli
from localagent.pipeline import flow


def test_train_cli_passes_resume_without_editing_config(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        flow,
        "run_stage",
        lambda stage, config, *, resume=False: calls.append((stage, config, resume)),
    )

    cli._train(
        SimpleNamespace(
            stage="sft",
            config="configs/train/sft-paper-tier-1m.yaml",
            resume=True,
        )
    )

    assert calls == [
        (
            "sft",
            "configs/train/sft-paper-tier-1m.yaml",
            True,
        )
    ]


def test_pipeline_resume_override_is_limited_to_exact_resume_stages(monkeypatch) -> None:
    called = []

    class _Module:
        @staticmethod
        def run(config_path: str, *, resume: bool = False) -> None:
            called.append((config_path, resume))

    monkeypatch.setattr("importlib.import_module", lambda _name: _Module)

    flow.run_stage("rl", "rl.yaml", resume=True)
    assert called == [("rl.yaml", True)]

    with pytest.raises(SystemExit, match="does not support exact resume"):
        flow.run_stage("eval", "eval.yaml", resume=True)


def test_sft_resume_git_receipt_requires_resume_and_routes_only_to_sft(monkeypatch) -> None:
    calls = []

    class _Module:
        @staticmethod
        def run(
            config_path: str,
            *,
            resume: bool = False,
            resume_git_receipt: str | None = None,
        ) -> None:
            calls.append((config_path, resume, resume_git_receipt))

    monkeypatch.setattr("importlib.import_module", lambda _name: _Module)
    flow.run_stage(
        "sft",
        "sft.yaml",
        resume=True,
        resume_git_receipt="/private/tmp/sft-receipt.json",
    )
    assert calls == [
        ("sft.yaml", True, "/private/tmp/sft-receipt.json"),
    ]
    with pytest.raises(SystemExit, match="requires --resume"):
        flow.run_stage(
            "sft",
            "sft.yaml",
            resume_git_receipt="/private/tmp/sft-receipt.json",
        )
    with pytest.raises(SystemExit, match="only the sft stage"):
        flow.run_stage(
            "rl",
            "rl.yaml",
            resume=True,
            resume_git_receipt="/private/tmp/sft-receipt.json",
        )


def test_train_cli_passes_sft_resume_git_receipt(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        flow,
        "run_stage",
        lambda stage, config, *, resume=False, resume_git_receipt=None: calls.append(
            (stage, config, resume, resume_git_receipt)
        ),
    )
    cli._train(
        SimpleNamespace(
            stage="sft",
            config="sft.yaml",
            resume=True,
            resume_git_receipt="/private/tmp/sft-receipt.json",
        )
    )
    assert calls == [
        ("sft", "sft.yaml", True, "/private/tmp/sft-receipt.json"),
    ]


def test_create_resume_git_receipt_cli_routes_reason_and_evidence(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "localagent.train.sft.create_resume_git_receipt",
        lambda config, out, *, reason, evidence: calls.append(
            (config, out, reason, evidence)
        ),
    )
    cli._create_resume_git_receipt(
        SimpleNamespace(
            config="sft.yaml",
            out="/private/tmp/sft-receipt.json",
            reason="non-numerical resume startup optimization",
            evidence=["focused tests passed", "reviewed diff"],
        )
    )
    assert calls == [
        (
            "sft.yaml",
            "/private/tmp/sft-receipt.json",
            "non-numerical resume startup optimization",
            ["focused tests passed", "reviewed diff"],
        )
    ]

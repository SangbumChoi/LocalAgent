#!/usr/bin/env python3
"""Run a public-train-only mobile/browser/desktop continuation and weight audit.

Each ``--train-data``/``--eval-data`` argument is ``LABEL=PATH`` and must already be a
normalized ``Conversation`` JSONL.  The script keeps source rows grouped in the receipt,
checks the combined train/eval split, reports per-surface teacher-forced metrics, and records
relative movement of the transferred backbone.  It deliberately does not claim native emulator,
browser, desktop, or external-account success.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.sft import _evaluate_conversations, sft
from scripts.analyze_weight_transfer import analyze as analyze_weight_transfer
from scripts.train_public_agent_continuation import (
    _assert_disjoint,
    _checkpoint_tokenizer,
    _identity,
    _load_rows,
)


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(raw_path).expanduser()


def _parse_source_reference(value: str) -> tuple[str, dict[str, str]]:
    label, separator, raw_reference = value.partition("=")
    dataset, url_separator, url = raw_reference.partition("|")
    if not separator or not label.strip() or not dataset.strip() or not url_separator or not url.strip():
        raise argparse.ArgumentTypeError("expected LABEL=DATASET|URL")
    return label.strip(), {"dataset": dataset.strip(), "url": url.strip()}


def _source_profile(
    label: str,
    path: Path,
    rows: list[Conversation],
    reference: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    datasets: set[str] = set()
    revisions: set[str] = set()
    splits: set[str] = set()
    source_families: set[str] = set()
    source_urls: set[str] = set()
    parent_ids: set[str] = set()
    omitted_visuals = 0
    for row in rows:
        meta = row.meta if isinstance(row.meta, Mapping) else {}
        provenance = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
        for key, values in (
            ("dataset", (provenance.get("dataset"), meta.get("source_dataset"))),
            ("revision", (provenance.get("revision"), meta.get("source_revision"))),
            ("split", (provenance.get("subset"), meta.get("source_split"), meta.get("split"))),
        ):
            target = datasets if key == "dataset" else revisions if key == "revision" else splits
            target.update(str(value) for value in values if value not in (None, ""))
        source_families.update(
            str(value)
            for value in (meta.get("source_family"), meta.get("source_dataset"))
            if value not in (None, "")
        )
        source_urls.update(
            str(value)
            for value in (provenance.get("url"), meta.get("source_url"))
            if value not in (None, "")
        )
        if meta.get("parent_record_id"):
            parent_ids.add(str(meta["parent_record_id"]))
        omitted_visuals += int(bool(meta.get("visual_input_omitted")))
    return {
        "label": label,
        "input": _identity(path),
        "rows": len(rows),
        "datasets": sorted(datasets),
        "revisions": sorted(revisions),
        "splits": sorted(splits),
        "source_families": sorted(source_families),
        "source_urls": sorted(source_urls),
        "public_reference": dict(reference) if reference else None,
        "unique_parent_records": len(parent_ids),
        "visual_input_omitted_rows": omitted_visuals,
    }


def _evaluate_by_source(
    grouped_rows: list[tuple[str, Path, list[Conversation]]],
    model: LocalAgentLM,
    tokenizer: Any,
    *,
    max_seq_len: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    return {
        label: _evaluate_conversations(
            model,
            rows,
            tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
        )
        for label, _path, rows in grouped_rows
    }


def _load_labeled_groups(values: Iterable[tuple[str, Path]]) -> list[tuple[str, Path, list[Conversation]]]:
    groups: list[tuple[str, Path, list[Conversation]]] = []
    labels: set[str] = set()
    for label, path in values:
        if label in labels:
            raise ValueError(f"duplicate source label: {label!r}")
        labels.add(label)
        rows = _load_rows([path])
        groups.append((label, path, rows))
    if not groups:
        raise ValueError("at least one labeled source is required")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument("--eval-data", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument(
        "--source-reference",
        type=_parse_source_reference,
        action="append",
        default=[],
        help="public source identity as LABEL=DATASET|URL; repeat for every source",
    )
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite continuation outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps and batch-size must be positive; lr must be positive")

    train_groups = _load_labeled_groups(args.train_data)
    eval_groups = _load_labeled_groups(args.eval_data)
    references = dict(args.source_reference)
    expected_labels = {label for label, _path, _rows in train_groups} | {
        label for label, _path, _rows in eval_groups
    }
    if set(references) != expected_labels:
        missing = sorted(expected_labels - set(references))
        extra = sorted(set(references) - expected_labels)
        raise SystemExit(f"source references must match labels; missing={missing}, extra={extra}")
    train_rows = [row for _label, _path, rows in train_groups for row in rows]
    eval_rows = [row for _label, _path, rows in eval_groups for row in rows]
    _assert_disjoint(train_rows, eval_rows)
    parent_identity = _identity(args.init)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    tokenizer = _checkpoint_tokenizer(parent)

    before_all = _evaluate_conversations(
        model, train_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    before_eval = _evaluate_conversations(
        model, eval_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    before_by_source = _evaluate_by_source(
        eval_groups, model, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    loss_history, _, _, training = sft(
        model,
        [],
        tokenizer,
        conversations=train_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=max(1, min(4, args.steps // 2)),
        device=args.device,
        max_seq_len=args.max_seq_len,
        seed=2027,
        log=print,
        return_metrics=True,
    )
    after_all = _evaluate_conversations(
        model, train_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    after_eval = _evaluate_conversations(
        model, eval_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    after_by_source = _evaluate_by_source(
        eval_groups, model, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )

    child = dict(parent)
    child.update(
        {
            "state_dict": model.state_dict(),
            "stage": "sft_cross_surface_public_continuation",
            "parent_checkpoint_sha256": parent_identity["sha256"],
            "cross_surface_training": {
                "train_sources": [
                    _source_profile(label, path, rows, references[label])
                    for label, path, rows in train_groups
                ],
                "eval_sources": [
                    _source_profile(label, path, rows, references[label])
                    for label, path, rows in eval_groups
                ],
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_seq_len": args.max_seq_len,
                "seed": 2027,
                "before_eval_by_source": before_by_source,
                "after_eval_by_source": after_by_source,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    transfer = analyze_weight_transfer(args.init, args.output)
    report = {
        "kind": "localagent_cross_surface_public_continuation_report",
        "schema_version": 1,
        "parent": parent_identity,
        "child": _identity(args.output),
        "train_sources": [
            _source_profile(label, path, rows, references[label])
            for label, path, rows in train_groups
        ],
        "eval_sources": [
            _source_profile(label, path, rows, references[label])
            for label, path, rows in eval_groups
        ],
        "rows": {"train": len(train_rows), "eval": len(eval_rows)},
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "seed": 2027,
            "device": args.device,
        },
        "before": {"train": before_all, "eval": before_eval, "eval_by_source": before_by_source},
        "after": {"train": after_all, "eval": after_eval, "eval_by_source": after_by_source},
        "loss_history": loss_history,
        "token_accounting": training,
        "weight_transfer": {
            "compatibility": transfer["compatibility"],
            "groups": transfer["groups"],
            "recommendation": transfer["recommendation"],
        },
        "claim_boundary": (
            "Public-train-only text/accessibility continuation across mobile, browser, and desktop "
            "projections with source-disjoint held-out metrics; no official benchmark score, native "
            "emulator/browser/desktop/MCP execution, screenshot grounding, or external-account claim."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

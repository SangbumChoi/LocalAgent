#!/usr/bin/env python3
"""Assemble a hash-bound matched warm/random ToolACE continuation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.data.toolace import (
    TOOLACE_ADAPTER_VERSION,
    TOOLACE_DATASET,
    TOOLACE_LICENSE,
    TOOLACE_REVISION,
    TOOLACE_URL,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _after(report: dict[str, Any]) -> float:
    return float(report["after"]["eval"]["assistant_token_accuracy"])


def _before(report: dict[str, Any]) -> float:
    return float(report["before"]["eval"]["assistant_token_accuracy"])


def _group_movement(report: dict[str, Any]) -> dict[str, float]:
    groups = report["weight_transfer"]["groups"]
    return {
        name: float(groups[name]["relative_delta_l2"])
        for name in ("embedding", "attention_or_mixer", "ffn", "normalization", "action_heads")
    }


def assemble(
    *,
    manifest_path: Path,
    warm_report_path: Path,
    random_report_path: Path,
    train_path: Path,
    eval_path: Path,
    warm_child_path: Path,
    random_child_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = _load(manifest_path)
    warm = _load(warm_report_path)
    random = _load(random_report_path)
    if manifest["dataset"] != TOOLACE_DATASET or manifest["revision"] != TOOLACE_REVISION:
        raise ValueError("ToolACE manifest identity mismatch")
    if warm["rows"] != random["rows"] or warm["parent"] != random["parent"]:
        raise ValueError("warm/random reports are not matched")
    if warm["eval_sources"][0]["input"] != random["eval_sources"][0]["input"]:
        raise ValueError("warm/random eval input mismatch")
    if warm["train_sources"][0]["input"] != random["train_sources"][0]["input"]:
        raise ValueError("warm/random train input mismatch")
    warm_after = _after(warm)
    random_after = _after(random)
    body: dict[str, Any] = {
        "kind": "localagent_current_child_toolace_transfer_receipt",
        "schema_version": 1,
        "measurement": "m171_current_child_toolace_first_action_transfer",
        "generated_at": "2026-08-03",
        "dataset": {
            "dataset": TOOLACE_DATASET,
            "revision": TOOLACE_REVISION,
            "url": TOOLACE_URL,
            "license": TOOLACE_LICENSE,
            "raw_source": manifest["source"],
            "projection_manifest": {
                "path": str(manifest_path),
                "self_sha256": manifest["manifest_self_sha256"],
                "adapter_version": TOOLACE_ADAPTER_VERSION,
                "raw_rows": manifest["raw_rows"],
                "accepted_rows": manifest["accepted_rows"],
                "rejected_rows": manifest["rejected_rows"],
                "rejections": manifest["rejections"],
                "full_train": manifest["outputs"]["train"],
                "full_eval": manifest["outputs"]["eval"],
                "split_audit": manifest["split_audit"],
            },
            "projection_boundary": manifest["projection"],
        },
        "parent": warm["parent"],
        "bounded_arm": {
            "train_rows": warm["rows"]["train"],
            "eval_rows": warm["rows"]["eval"],
            "train_input": warm["train_sources"][0]["input"],
            "eval_input": warm["eval_sources"][0]["input"],
            "steps": warm["hyperparameters"]["steps"],
            "batch_size": warm["hyperparameters"]["batch_size"],
            "learning_rate": warm["hyperparameters"]["learning_rate"],
            "max_seq_len": warm["hyperparameters"]["max_seq_len"],
            "device": warm["hyperparameters"]["device"],
            "seed": warm["hyperparameters"]["seed"],
        },
        "comparison": {
            "warm_start": {
                "before_token_accuracy": _before(warm),
                "after_token_accuracy": warm_after,
                "delta_token_accuracy": warm_after - _before(warm),
                "after_mean_loss": warm["after"]["eval"]["mean_loss"],
                "sequence_accuracy": warm["after"]["eval"]["assistant_sequence_accuracy"],
                "child": {**warm["child"], "path": str(warm_child_path)},
                "backbone_movement": _group_movement(warm),
            },
            "random_backbone": {
                "before_token_accuracy": _before(random),
                "after_token_accuracy": random_after,
                "delta_token_accuracy": random_after - _before(random),
                "after_mean_loss": random["after"]["eval"]["mean_loss"],
                "sequence_accuracy": random["after"]["eval"]["assistant_sequence_accuracy"],
                "child": {**random["child"], "path": str(random_child_path)},
                "backbone_movement": _group_movement(random),
            },
            "warm_minus_random_after_pp": (warm_after - random_after) * 100.0,
            "warm_start_better_after": warm_after > random_after,
        },
        "compatibility": warm["weight_transfer"]["compatibility"],
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Source-record-disjoint public ToolACE first-action projection with a bounded 1,024/256 "
            "continuation and matched random-backbone control. This is not an official ToolACE or "
            "BFCL split/score, not a multi-turn execution result, and not native mobile, browser, "
            "desktop, email, Notion, MCP, or external-account success."
        ),
        "publication": {
            "public_hub_url": None,
            "uploaded": False,
            "reason": "Hugging Face authentication is not configured.",
        },
    }
    body["receipt_self_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(body) + b"\n")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--warm-child", type=Path, required=True)
    parser.add_argument("--random-child", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(
                manifest_path=args.manifest,
                warm_report_path=args.warm_report,
                random_report_path=args.random_report,
                train_path=args.train_data,
                eval_path=args.eval_data,
                warm_child_path=args.warm_child,
                random_child_path=args.random_child,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

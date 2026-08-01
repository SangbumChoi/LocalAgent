#!/usr/bin/env python
"""Normalize an acquired AgentNet JSONL snapshot for offline evaluation.

The command never downloads AgentNet and never accepts a training split.  The input must be a
locally acquired JSONL file whose full byte identity is recorded in the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from localagent.data.agentnet import AGENTNET_LICENSE, AGENTNET_REVISION, AGENTNET_URL
from localagent.data.agentnet import normalize_agentnet_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision", default=AGENTNET_REVISION)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    if args.input.resolve() in {args.output.resolve(), args.manifest.resolve()}:
        raise SystemExit("output and manifest must differ from input")
    if args.max_records < 0:
        raise SystemExit("max-records must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    records = 0
    with args.input.open("r", encoding="utf-8", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for source_line, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                normalized = normalize_agentnet_record(
                    raw,
                    source_revision=args.revision,
                    split="eval",
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise SystemExit(f"line {source_line}: {error}") from error
            encoded = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            destination.write(encoded.decode("utf-8"))
            output_digest.update(encoded)
            records += 1
            if args.max_records and records >= args.max_records:
                break
    input_bytes = args.input.stat().st_size
    input_sha = _sha256(args.input)
    manifest: dict[str, object] = {
        "kind": "localagent_agentnet_acquisition_manifest",
        "schema_version": 1,
        "dataset": "AgentNet",
        "dataset_url": AGENTNET_URL,
        "dataset_license": AGENTNET_LICENSE,
        "source_revision": args.revision,
        "source": {
            "path": str(args.input),
            "bytes": input_bytes,
            "sha256": input_sha,
            "split": "eval",
        },
        "normalization": {
            "module": "localagent.data.agentnet",
            "version": 1,
            "interchange": "localagent_v1",
            "observation_policy": "text_observation_required; screenshots are not consumed",
            "action_policy": "preserve_agentnet_coordinate_actions_for_offline_scoring",
        },
        "records": {
            "selected": records,
            "output_path": str(args.output),
            "output_sha256": output_digest.hexdigest(),
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

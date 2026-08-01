#!/usr/bin/env python
"""Verify or stage a generated LocalAgent WebGPU demo bundle."""

from __future__ import annotations

import argparse
import json
import sys

from localagent.eval.demo_deploy import (
    sync_demo_bundle,
    verify_demo_deploy,
    write_demo_deploy_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", default="spaces/localagent-webgpu")
    parser.add_argument("--bundle-dir", help="generated export directory; defaults to --demo-dir")
    parser.add_argument("--sync", action="store_true", help="copy a verified bundle beside the app")
    parser.add_argument("--out", help="write a new JSON receipt (refuses overwrite)")
    args = parser.parse_args()

    if args.sync:
        if not args.bundle_dir:
            parser.error("--sync requires --bundle-dir")
        report = sync_demo_bundle(args.bundle_dir, args.demo_dir)
    else:
        report = verify_demo_deploy(args.demo_dir, bundle_dir=args.bundle_dir)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out:
        write_demo_deploy_receipt(report, args.out)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

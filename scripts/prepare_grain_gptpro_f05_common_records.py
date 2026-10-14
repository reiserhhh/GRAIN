#!/usr/bin/env python3
"""Convert and verify every frozen F05 source/validation/application record."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256, validate_record
from grain_research.gptpro_foundation_v3.f05_records import from_common, to_common


def main():
    """Save a new compressed view; preserve originals and verify exact reversal."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior conversions")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.source / "output_manifest.json").read_text())
    for item in manifest["files"]:
        assert sha256(args.source / item["path"]) == item["sha256"]
    source = json.loads((args.source / "summary.json").read_text())
    outputs = []
    total = 0
    for outcome in source["outcomes"]:
        name = f"{outcome['world']}_{outcome['seed']}"
        root = args.source / name
        target = args.output / name
        target.mkdir()
        splitmap = {}
        seen = set()
        for role in ("source", "validation", "application"):
            path = root / f"{role}.jsonl"
            count = 0
            destination = target / f"{role}.jsonl.gz"
            with path.open() as original, gzip.open(destination, "wt") as handle:
                for line in original:
                    row = json.loads(line)
                    author = row["author_id"]
                    if author in splitmap and splitmap[author] != role:
                        raise ValueError("author split overlap")
                    splitmap[author] = role
                    common = to_common(row)
                    validate_record(common, splitmap, seen)
                    encoded = json.dumps(common, allow_nan=False)
                    assert from_common(json.loads(encoded)) == row
                    handle.write(encoded + "\n")
                    count += 1
            # Read the actual compressed artifact rather than only in-memory data.
            with gzip.open(destination, "rt") as handle, path.open() as original:
                for left, right in zip(handle, original, strict=True):
                    assert from_common(json.loads(left)) == json.loads(right)
            total += count
            outputs.append(
                {
                    "bundle": name,
                    "role": role,
                    "records": count,
                    "input_sha256": sha256(path),
                    "output_sha256": sha256(destination),
                }
            )
        (target / "splitmap.json").write_text(json.dumps(splitmap, indent=2) + "\n")
    summary = {
        "status": "COMMON_RECORD_ROUNDTRIP_VERIFIED",
        "records": total,
        "files": outputs,
        "feature_boundary": "estimators receive z/y/t through explicit adapter, never metadata IDs or world",
        "scope": "data-contract conversion, not new scoring or scientific acceptance",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "source_adapter.py").write_bytes(
        (ROOT / "grain_research/gptpro_foundation_v3/f05_records.py").read_bytes()
    )
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {"status": summary["status"], "records": total, "role_files": len(outputs)}
        )
    )


if __name__ == "__main__":
    main()

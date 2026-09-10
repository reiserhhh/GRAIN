#!/usr/bin/env python3
"""Publish audited NEXT71 aggregates with private inputs/models left in place."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_evidence_components import OUT, cache, old


def publish():
    """Copy the six declared aggregate tables and their provenance."""
    m = cache.manifest(OUT)
    audit, summary = (
        old.read(OUT / "audit.json"),
        old.read(OUT / "analysis/summary.json"),
    )
    if audit["status"] != "PASS":
        raise ValueError("Primary audit required")
    old.verify_hashes(OUT / "analysis", summary["files"])
    dst = ROOT / "reports/grain_vector_mechanisms/evidence_components_v1"
    dst.mkdir(exist_ok=True)
    if len(summary["files"]) != 6:
        raise ValueError("Six aggregate tables required")
    for name in summary["files"]:
        src = OUT / "analysis" / name
        if name.startswith("private_") or set(pd.read_csv(src, nrows=0).columns) & {
            "author_key",
            "text",
            "user_id",
        }:
            raise ValueError("Private data in aggregate")
        shutil.copyfile(src, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": summary["files"],
            "publisher_sha256": old.sha(Path(__file__)),
            "post_account_geometry_test_sha256": old.sha(
                ROOT / "tests/test_grain_evidence_component_geometry.py"
            ),
            "scope": "Frozen source4/new8 native-query function account; no causal or human validity shares",
        },
    )
    print({"status": "PUBLISHED", "tables": 6}, flush=True)


if __name__ == "__main__":
    publish()

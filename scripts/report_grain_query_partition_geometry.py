#!/usr/bin/env python3
"""Publish only audited author-aggregate NEXT72 moments and provenance."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_query_partition_geometry import OUT, old, table_hashes


def publish():
    """Copy three aggregate tables after source, cardinality and replay checks."""
    m = old.verify(OUT)
    audit = old.read(OUT / "audit.json")
    s = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS" or s["counts"] != [24, 96, 12]:
        raise ValueError("Complete audited geometry required")
    old.verify_hashes(OUT / "analysis", s["files"])
    if table_hashes(OUT.parent / "analysis") != s["files"]:
        raise ValueError("Inventory correction unexpectedly changed numerical tables")
    dst = ROOT / "reports/grain_vector_mechanisms/query_partition_geometry_v1"
    dst.mkdir(exist_ok=True)
    for name in s["files"]:
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {
            "author_key",
            "text",
            "user_id",
            "head",
            "token",
        }:
            raise ValueError("Private data in aggregate")
        shutil.copyfile(src, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": s["files"],
            "publisher_sha256": old.sha(Path(__file__)),
            "initial_numeric_tables_unchanged": True,
            "inventory_amendment": "R2 excludes AppleDouble sidecars; initial packet retained",
            "scope": "Source4/new8 frozen native key actions; signed joint geometry, not human personality accuracy",
        },
    )
    print({"status": "PUBLISHED", "tables": len(s["files"])}, flush=True)


if __name__ == "__main__":
    publish()

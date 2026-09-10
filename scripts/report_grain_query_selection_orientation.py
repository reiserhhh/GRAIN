#!/usr/bin/env python3
"""Publish only the five audited NEXT73 aggregate tables."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_query_selection_orientation import OUT, TABLES, old


def publish():
    """Check the exact inventory and publish no author/token/head records."""
    m = old.verify(OUT)
    audit, s = old.read(OUT / "audit.json"), old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS" or set(s["files"]) != set(TABLES):
        raise ValueError("Exact audited table set required")
    old.verify_hashes(OUT / "analysis", s["files"])
    dst = ROOT / "reports/grain_vector_mechanisms/query_selection_orientation_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {
            "author_key",
            "user_id",
            "text",
            "token",
            "head",
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
            "post_account_theory_test_sha256": old.sha(
                ROOT / "tests/test_grain_query_selection_orientation_theory.py"
            ),
            "scope": "Frozen source-query crossed scalar paths, symmetric swap attribution, not human personality shares",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

#!/usr/bin/env python3
"""Publish audited source endpoint aggregates, never material or source readers."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_endpoint_assembly import OUT, TABLES, freeze, old


def publish():
    """Copy the exact finite inventory and its provenance after successful replay."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    f = freeze(OUT, "analysis")
    if a["status"] != "PASS" or set(f["files"]) != set(TABLES):
        raise ValueError("Complete audited table inventory required")
    dst = ROOT / "reports/grain_vector_mechanisms/source_endpoint_assembly_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        if {"author_key", "user_id", "text", "token", "head", "reference"} & set(
            pd.read_csv(src, nrows=0).columns
        ):
            raise ValueError("Private fields cannot be published")
        shutil.copyfile(src, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": a,
            "files": {n: old.sha(dst / n) for n in TABLES},
            "publisher_sha256": old.sha(Path(__file__)),
            "scope": "H16 plus source-only selector, value and endpoint maps. K+V fixed-Q component only; no current gradients/native values, no full L or questionnaire prediction.",
            "cpu_scope": "Successful numeric stages including replay; provenance preflight and imports excluded.",
            "reference": "source4 leave-one-author-out; new8 all4 source reference. Reused discovery materials, not a fresh blind sample.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

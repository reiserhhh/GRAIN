#!/usr/bin/env python3
"""Publish NEXT78 aggregate mechanism evidence, excluding private term packets."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_endpoint_query_contrast import OUT, TABLES, old, previous


def publish():
    """Copy only the canonical five tables after exact replay and source checks."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    f = previous.freeze(OUT, "analysis")
    if a["status"] != "PASS" or not set(TABLES) <= set(f["files"]):
        raise ValueError("Audited complete aggregate inventory required")
    dst = ROOT / "reports/grain_vector_mechanisms/endpoint_query_contrast_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        if {"author_key", "user_id", "text", "token", "reference"} & set(
            pd.read_csv(src, nrows=0).columns
        ):
            raise ValueError("Private fields prohibited in public tables")
        shutil.copyfile(src, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": a,
            "files": {n: old.sha(dst / n) for n in TABLES},
            "publisher_sha256": old.sha(Path(__file__)),
            "scope": "Fixed two-query endpoint component contrast, not full L/personality scores. Author centering is descriptive only; signed covariance projections are not causal percentages.",
            "cpu_scope": "Successful numeric stages including replay; provenance preflight and imports excluded.",
            "reference": "source4LOO/new8all4 reused discovery panel; four GRAIN columns are not additional independent authors.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

#!/usr/bin/env python3
"""Publish only audited aggregate source-motion and endpoint comparisons."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_reader_motion import OUT, TABLES, ep, old


def publish():
    """Keep source perturbations, context arrays and native reader packets private."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    f = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or not set(TABLES) <= set(f["files"]):
        raise ValueError("Audited complete results required")
    dst = ROOT / "reports/grain_vector_mechanisms/source_reader_motion_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        t = pd.read_csv(src)
        if {"author_key", "user_id", "text", "token", "reference"} & set(t.columns):
            raise ValueError("Private fields forbidden")
        if name == "source_motion_summary.csv":
            if set(t.reference_design) != {"source3_LOO", "source4_all"}:
                raise ValueError("Unexpected source-only aggregation")
        elif set(t.cohort) != {"source4", "new8"}:
            raise ValueError("Unexpected author cohort")
        shutil.copyfile(src, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "runtime_amendments": m.get("runtime_amendments", []),
            "audit": a,
            "files": {n: old.sha(dst / n) for n in TABLES},
            "publisher_sha256": old.sha(Path(__file__)),
            "scope": "Source H16-last interventions, post-attention22 reader motion and fixed endpoint components. Not natural text edits, full L replacement or human personality scores.",
            "reference": "source4LOO/new8all4 reused discovery panels; source/reference pairs and GRAIN factors are not additional authors.",
            "cpu_scope": "Successful canonical numeric stages including cached replay; native time and loading separate. First failed evaluation CPU was not captured and is excluded, not zero; its artifacts are preserved. Hashing preflight, tests and publication excluded.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

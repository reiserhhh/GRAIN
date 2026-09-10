#!/usr/bin/env python3
"""Publish only audited cohort aggregates for the conditional reader experiment."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_context_differential_reader import OUT, TABLES, ep, old


def publish():
    """Keep contexts, source weights, identities and reader arrays private."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    f = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or not set(TABLES) <= set(f["files"]):
        raise ValueError("Complete audited aggregate inventory required")
    dst = ROOT / "reports/grain_vector_mechanisms/context_differential_reader_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        t = pd.read_csv(src)
        if {"author_key", "user_id", "text", "token", "reference"} & set(t.columns):
            raise ValueError("Private columns forbidden")
        if set(t.cohort) != {"source4", "new8"}:
            raise ValueError("Only declared cohort aggregates may be published")
        shutil.copyfile(src, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": a,
            "files": {n: old.sha(dst / n) for n in TABLES},
            "publisher_sha256": old.sha(Path(__file__)),
            "scope": "Fixed local endpoint scalars and GRAIN BODY directions, not full L or personality scores. Source affine bound is a vector-error bound, not an action-error bound.",
            "reference": "source4LOO/new8all4 reused discovery materials; factor columns are not independent authors.",
            "cpu_scope": "Successful numeric stages including replay; provenance preflight and imports excluded.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

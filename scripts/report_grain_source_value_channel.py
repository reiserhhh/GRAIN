#!/usr/bin/env python3
"""Publish audited NEXT76 aggregates without private material or model arrays."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_value_channel import OUT, TABLES, old, prior


def publish():
    """Export exact canonical tables, not filesystem sidecars or author-level rows."""
    m = old.verify(OUT)
    a = old.read(OUT / "audit.json")
    f = prior.freeze(OUT, "analysis")
    prior.freeze(OUT, "models")
    if a["status"] != "PASS" or set(f["files"]) != set(TABLES):
        raise ValueError("Audited complete canonical analysis required")
    dst = ROOT / "reports/grain_vector_mechanisms/source_value_channel_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        if {"author_key", "user_id", "text", "token", "head", "reference"} & set(
            pd.read_csv(src, nrows=0).columns
        ):
            raise ValueError("Private fields in aggregate export")
        shutil.copyfile(src, dst / name)
    tr = pd.read_csv(OUT / "models/source_training.csv").drop(columns="reference")
    tr.groupby(["source_authors", "query"], as_index=False).mean().to_csv(
        dst / "source_training.csv", index=False
    )
    names = (*TABLES, "source_training.csv")
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": a,
            "files": {n: old.sha(dst / n) for n in names},
            "publisher_sha256": old.sha(Path(__file__)),
            "scope": "H16 plus source Q/value maps and retained fixed g. Totals are key+value with Q/g fixed, not full L or human questionnaire scores.",
            "cpu_scope": "Successful numeric stages including replay; provenance preflight and imports excluded.",
            "source_summary": "source_authors=3 averages four LOO fits; 4 denotes one all-source reference",
        },
    )
    print({"status": "PUBLISHED", "tables": len(names)}, flush=True)


if __name__ == "__main__":
    publish()

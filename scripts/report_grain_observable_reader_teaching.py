#!/usr/bin/env python3
"""Publish only audited source/new8 cohort summaries of functional reader teaching."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_observable_reader_teaching import COUNTS, OUT, TABLES, ep, old


def publish():
    """Preserve all fixed comparisons without releasing reader vectors or identities."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    f = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or f["counts"] != COUNTS:
        raise ValueError("Complete audited comparison required")
    dst = ROOT / "reports/grain_vector_mechanisms/observable_reader_teaching_v1"
    dst.mkdir(exist_ok=True)
    for name, count in zip(TABLES, COUNTS, strict=True):
        t = pd.read_csv(OUT / "analysis" / name)
        if len(t) != count or not set(t.authors) <= {4, 8}:
            raise ValueError("Cohort summary inventory differs")
        if {"author_key", "user_id", "text", "token", "reference"} & set(t):
            raise ValueError("Private fields forbidden")
        shutil.copyfile(OUT / "analysis" / name, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "publisher_sha256": old.sha(Path(__file__)),
            "files": {n: old.sha(dst / n) for n in TABLES},
            "audit": a,
            "scope": "Source4ALL teaches two fixed readers; new8 is reused discovery material. H16 remains native. No new human labels, clinical criteria, neural calls or current target reads in application.",
            "range_account": "Post-freeze native-reader output-range explanation only; not an oracle predictor. Scalar projection cross-terms retained.",
            "cpu_scope": "Canonical preparation, source fits, application/evaluation and cached replay; excludes later preflight hashing, tests, publisher, audit and terminal exploratory summaries.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

#!/usr/bin/env python3
"""Publish the complete NEXT84 cohort comparison, without source or reader arrays."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_observable_span import COUNTS, OUT, TABLES, ep, old


def publish():
    """Require frozen audited tables and retain all ten fixed comparators."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    fr = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or fr["counts"] != COUNTS:
        raise ValueError("Complete audited comparison required")
    dst = ROOT / "reports/grain_vector_mechanisms/source_observable_span_v1"
    dst.mkdir(exist_ok=True)
    for n, count in zip(TABLES, COUNTS, strict=True):
        t = pd.read_csv(OUT / "analysis" / n)
        if len(t) != count or not set(t.authors) <= {4, 8}:
            raise ValueError("Cohort inventory differs")
        if {"author_key", "user_id", "text", "token", "reference"} & set(t):
            raise ValueError("Private columns forbidden")
        shutil.copyfile(OUT / "analysis" / n, dst / n)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "publisher_sha256": old.sha(Path(__file__)),
            "files": {n: old.sha(dst / n) for n in TABLES},
            "audit": a,
            "scope": "Source4ALL, same28 observations and20 unique source rulers; new8 reused discovery panel. H16 remains native. Zero new human labels, neural calls or API calls.",
            "range_account": "Post-freeze signed native-error explanation, never an oracle predictor. Old eight comparator statistics retained.",
            "prior_beta_relative": "For new fields, versus frozen U0 same-objective fit; historical rows retain their original comparison.",
            "cpu_scope": "Canonical preparation, models, application, evaluation and cached replay; excludes preflight hashing, tests, audit, publisher and later exploratory summaries.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

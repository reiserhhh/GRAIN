#!/usr/bin/env python3
"""Publish audited cohort aggregates for the conditional teaching experiment."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_conditional_teaching import COUNTS, OUT, TABLES, ep, old


def publish():
    """Copy only the fixed public tables, with hashes and execution provenance."""
    m, audit = old.verify(OUT), old.read(OUT / "audit.json")
    fr = ep.freeze(OUT, "analysis")
    if audit["status"] != "PASS" or fr["counts"] != COUNTS:
        raise ValueError("Complete audited analysis required")
    dst = ROOT / "reports/grain_vector_mechanisms/conditional_teaching_v1"
    dst.mkdir(exist_ok=False)
    for name, count in zip(TABLES, COUNTS, strict=True):
        table = pd.read_csv(OUT / "analysis" / name)
        if len(table) != count or not set(table.authors) <= {4, 8}:
            raise ValueError("Cohort inventory differs")
        if {"author_key", "user_id", "text", "token", "reference"} & set(table):
            raise ValueError("Private columns forbidden")
        shutil.copyfile(OUT / "analysis" / name, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "publisher_sha256": old.sha(Path(__file__)),
            "files": {n: old.sha(dst / n) for n in TABLES},
            "audit": audit,
            "scope": "Source4ALL, reused new8 discovery material; six nested functions, fixed kernel/lambda/group weights. Native H16 and compiled frames retained. No new neural/API/human-label/MEPS use.",
            "conditioning": "V -> VB -> VBC; intermediate functions are conditional solves, not raw coefficient-group deletion. Full VBC matches NEXT85.",
            "cpu_scope": "Canonical preparation/models/application/evaluation/exact replay, excluding preflight hashes/tests/audit/publication.",
            "comparators": "All twelve NEXT85 readers remain in the six core tables; six new nested functions appended.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

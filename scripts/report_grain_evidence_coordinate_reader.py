#!/usr/bin/env python3
"""Publish only audited cohort-level NEXT85 comparisons and numerical geometry."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_evidence_coordinate_reader import COUNTS, OUT, TABLES, ep, old


def publish():
    """Retain all prior comparisons, excluding author-level inputs and outputs."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    fr = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or fr["counts"] != COUNTS:
        raise ValueError("Complete audited analysis required")
    dst = ROOT / "reports/grain_vector_mechanisms/evidence_coordinate_reader_v1"
    dst.mkdir(exist_ok=True)
    for n, count in zip(TABLES, COUNTS, strict=True):
        table = pd.read_csv(OUT / "analysis" / n)
        if len(table) != count or not set(table.authors) <= {4, 8}:
            raise ValueError("Cohort inventory differs")
        if {"author_key", "user_id", "text", "token", "reference"} & set(table):
            raise ValueError("Private columns forbidden")
        shutil.copyfile(OUT / "analysis" / n, dst / n)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "publisher_sha256": old.sha(Path(__file__)),
            "files": {n: old.sha(dst / n) for n in TABLES},
            "audit": a,
            "scope": "Four source authors teach fixed/moving coordinate functions; new8 is reused discovery material. Current frames use the all4 source-compiled program and H16. No new native neural calls, API calls or human labels.",
            "teacher_binding": "Eight original/clamped values, sixteen clamped BODY derivatives, four original context derivatives. Moving targets include the frame product term.",
            "cpu_scope": "Canonical preparation, teachers, models, application, analysis and exact cached replay; excludes preflight hashing, tests, audit, publishing and exploratory summaries.",
            "table_scope": "Old ten-reader tables preserved; new coordinate_fit/frame_geometry/coordinate_prediction/lift_checks/correction/derivative tables describe NEXT85.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

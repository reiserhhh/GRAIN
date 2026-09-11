#!/usr/bin/env python3
"""Export aggregate joint-query results without author identities or vectors."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_joint_query_functional_reader import COUNTS, OUT, TABLES, ep, old


def publish():
    """Publish only audited cohort tables and execution provenance."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    f = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or f["counts"] != COUNTS:
        raise ValueError("Audited full result required")
    dst = ROOT / "reports/grain_vector_mechanisms/joint_query_functional_reader_v1"
    dst.mkdir(exist_ok=False)
    for n, count in zip(TABLES, COUNTS, strict=True):
        t = pd.read_csv(OUT / "analysis" / n)
        if len(t) != count or not set(t.authors) <= {8, 12}:
            raise ValueError("Public cohort inventory differs")
        if {"author_id", "author_key", "user_id", "text", "token", "reference"} & set(
            t
        ):
            raise ValueError("Private columns forbidden")
        shutil.copyfile(OUT / "analysis" / n, dst / n)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "publisher_sha256": old.sha(Path(__file__)),
            "files": {n: old.sha(dst / n) for n in TABLES},
            "audit": a,
            "scope": "Source12, reused new8 evaluation-only. Local model components, not questionnaire or final LLM accuracy. H16 remains neural-derived.",
            "method": "One ten-coordinate source solve, fixed old input/row units/weight/lambda; per-query minimal lift and nonzero shared-reader derivative.",
            "comparators": "All22 old rows retained; paired mean and vector decomposition compare native12/compatible12/joint12.",
            "cpu_scope": "Canonical numeric construction/application/evaluation and exact replay; hashes/tests/publication excluded.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

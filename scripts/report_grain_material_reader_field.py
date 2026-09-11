#!/usr/bin/env python3
"""Publish only audited cohort summaries of source-learned material reader fields."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_material_reader_field import COUNTS, OUT, TABLES, ep, old


def publish():
    """Keep source teachers and recipient vectors private; publish all fixed arms."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    fr = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or fr["counts"] != COUNTS:
        raise ValueError("Audited complete results required")
    dst = ROOT / "reports/grain_vector_mechanisms/material_reader_field_v1"
    dst.mkdir(exist_ok=True)
    for name, count in zip(TABLES, COUNTS, strict=True):
        t = pd.read_csv(OUT / "analysis" / name)
        if len(t) != count or not set(t.authors) <= {4, 8}:
            raise ValueError("Unexpected aggregate population")
        if {"author_key", "user_id", "text", "token", "reference"} & set(t):
            raise ValueError("Private fields forbidden")
        shutil.copyfile(OUT / "analysis" / name, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "implementation_commit": "7fdf436",
            "pre_fit_test_fix": "f0a0494",
            "publisher_sha256": old.sha(Path(__file__)),
            "audit": a,
            "files": {n: old.sha(dst / n) for n in TABLES},
            "scope": "Source4ALL teaches 28 value/directional observations; new8 is a reused discovery panel. Native H16 remains. No current native targets in application, no human labels or clinical claims.",
            "derivatives": "Anchored H partials have the prior local-component target; diagonal derivatives include reader motion and are not scored against that mismatched target.",
            "cpu_scope": "Canonical preparation (including its preflight), source solves, application/evaluation and cached replay. Later preflight hashing, tests, publication and audit excluded. An earlier preparation before the test-only column-access fix is preserved separately, outside canonical timing; no model was fitted there.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

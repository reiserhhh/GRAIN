#!/usr/bin/env python3
"""Publish aggregate source12 reader comparisons without private author data."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_diversity_teaching import COUNTS, OUT, TABLES, ep, old


def publish():
    """Export audited aggregate tables and provenance only."""
    m, a = old.verify(OUT), old.read(OUT / "audit.json")
    fr = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or fr["counts"] != COUNTS:
        raise ValueError("Audited completed analysis required")
    dst = ROOT / "reports/grain_vector_mechanisms/source_diversity_teaching_v1"
    dst.mkdir(exist_ok=False)
    for name, count in zip(TABLES, COUNTS, strict=True):
        table = pd.read_csv(OUT / "analysis" / name)
        if len(table) != count or not set(table.authors) <= {8, 12}:
            raise ValueError("Cohort table differs")
        if {"author_key", "user_id", "text", "token", "reference", "author_id"} & set(
            table
        ):
            raise ValueError("Private columns forbidden")
        shutil.copyfile(OUT / "analysis" / name, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "publisher_sha256": old.sha(Path(__file__)),
            "files": {n: old.sha(dst / n) for n in TABLES},
            "audit": a,
            "scope": "Original4+disjoint8 sources; reused new8 evaluation-only. Local native component, not human trait labels or final LLM score. H16 remains neural-derived.",
            "comparators": "All20 prior readers preserved; matched12 native/compatible value teachers. Source units, row weight, lambda, upstream all4 programs fixed.",
            "native_scope": "16 full forward H16 hook captures,64 pre22 JVP,16 reverse pairs,1 independent endpoint JVP,4 suffix FD. Preparation and model loading separate.",
            "cpu_scope": "Canonical numeric stages including replay, excluding preparation/tests/publication; native wall separately capped.",
            "common_prefix": "Paired H16 common-prefix canonicalization follows NEXT37, with observed rounding difference retained privately.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

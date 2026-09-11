#!/usr/bin/env python3
"""Publish only aggregate matched-teaching results and execution provenance."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_compiled_compatible_reader import COUNTS, OUT, TABLES, ep, old


def publish():
    """Copy finite cohort tables, never native packets or author-level outputs."""
    m, audit = old.verify(OUT), old.read(OUT / "audit.json")
    fr = ep.freeze(OUT, "analysis")
    if audit["status"] != "PASS" or fr["counts"] != COUNTS:
        raise ValueError("Complete audited analysis required")
    dst = ROOT / "reports/grain_vector_mechanisms/compiled_compatible_reader_v1"
    dst.mkdir(exist_ok=False)
    for name, count in zip(TABLES, COUNTS, strict=True):
        table = pd.read_csv(OUT / "analysis" / name)
        if len(table) != count or not set(table.authors) <= {4, 8}:
            raise ValueError("Cohort table inventory differs")
        if {"author_key", "user_id", "text", "token", "reference"} & set(table):
            raise ValueError("Private columns forbidden")
        shutil.copyfile(OUT / "analysis" / name, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "prior_attempt": m["prior_attempt"],
            "publisher_sha256": old.sha(Path(__file__)),
            "files": {n: old.sha(dst / n) for n in TABLES},
            "audit": audit,
            "scope": "Source4ALL, reused new8 discovery material; two matched original4 targets, fixed Gaussian kernel/lambda/weights/row units. H16 remains neural-derived. Local component, not final LLM score or questionnaire accuracy.",
            "comparators": "All eighteen prior readers retained. Old8-value V is not the matched4-value control.",
            "cpu_scope": "Canonical numerical stages including exact replay, excluding tests/preflight hashes/publication.",
            "source_native": "Eight cached local-query reconstructions, not new neural calls. Recipient native evaluation targets opened only after prediction freeze.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

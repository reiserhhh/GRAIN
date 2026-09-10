#!/usr/bin/env python3
"""Export only audited source-cohort reader material dependence summaries."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_reader_material_dependence import COUNTS, OUT, TABLES, ep, old


def publish():
    """Publish factor-coded aggregate effects, excluding source states and identities."""
    m = old.verify(OUT)
    a = old.read(OUT / "audit.json")
    f = ep.freeze(OUT, "analysis")
    if a["status"] != "PASS" or f["counts"] != COUNTS:
        raise ValueError("Complete audited NEXT81 required")
    dst = ROOT / "reports/grain_vector_mechanisms/reader_material_dependence_v1"
    dst.mkdir(exist_ok=True)
    for name, count in zip(TABLES, COUNTS, strict=True):
        t = pd.read_csv(OUT / "analysis" / name)
        if len(t) != count or not (t.authors == 4).all():
            raise ValueError("Four-source aggregate inventory required")
        if {"author_key", "user_id", "text", "token", "reference"} & set(t):
            raise ValueError("Private fields forbidden")
        shutil.copyfile(OUT / "analysis" / name, dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "publisher_sha256": old.sha(Path(__file__)),
            "audit": a,
            "files": {n: old.sha(dst / n) for n in TABLES},
            "scope": "Four reused source authors, two fixed queries and four existing GRAIN directions; no current recipient experiment, human labels or clinical interpretation.",
            "projections": "Original native or frozen author-LOO numerical evidence vectors are fixed rulers, not clamped-state native target reconstructions.",
            "cpu_scope": "Successful canonical preparation/analysis/replay. Preparation includes its preflight. Native and loading separate; analysis preflight hashing, tests, publication and audits excluded.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

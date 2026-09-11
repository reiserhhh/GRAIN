#!/usr/bin/env python3
"""Publish only complete cohort-level current-moment findings."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_moment_nullspace as run


def publish():
    """Keep author identity, text, token scores and native fields in private storage."""
    m, audit = run.verify(run.OUT), run.old.read(run.OUT / "audit.json")
    if audit["status"] != "PASS":
        raise ValueError("Complete audit required")
    fr = run.old.read(run.OUT / "analysis/freeze.json")
    run.old.verify_hashes(run.OUT / "analysis", fr["files"])
    dst = ROOT / "reports/grain_vector_mechanisms/joint_moment_nullspace_v1"
    dst.mkdir(exist_ok=False)
    for name, count in zip(run.fn.TABLES, run.fn.COUNTS, strict=True):
        p = run.OUT / "analysis" / (name + ".csv")
        t = pd.read_csv(p)
        if (
            len(t) != count
            or set(t.authors) != {4, 8}
            or {"author_key", "text", "token", "user_id"} & set(t)
        ):
            raise ValueError("Aggregate-only complete cohort table required")
        shutil.copyfile(p, dst / p.name)
    run.old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": {
                n + ".csv": run.old.sha(dst / (n + ".csv")) for n in run.fn.TABLES
            },
            "publisher_sha256": run.old.sha(Path(__file__)),
            "scope": "Fixed original4 and reused8; cached native full-L H16 gradients. Not questionnaire validity, finite text edits, local reader c or proof that the entire compiled assembly is blind.",
            "metric": "Declared Frobenius coordinate metric and frozen GRAIN/address values; ratio-of-sums and per-author fractions reported separately.",
        },
    )
    print({"status": "PUBLISHED", "tables": len(run.fn.TABLES)}, flush=True)


if __name__ == "__main__":
    publish()

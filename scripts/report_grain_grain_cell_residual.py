#!/usr/bin/env python3
"""Publish only verified cohort-level NEXT91 tables, never private author records."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_grain_cell_residual as run


def publish():
    """Bind public aggregate tables to the completed experiment and replay audit."""
    run.verify(run.OUT)
    audit = run.prev.old.read(run.OUT / "audit.json")
    if audit["status"] != "PASS" or not audit["all_aggregates_rebuilt"]:
        raise ValueError("Completed aggregate audit required")
    stage = run.prev.old.read(run.OUT / "analysis/freeze.json")
    run.prev.old.verify_hashes(run.OUT / "analysis", stage["files"])
    out = ROOT / "reports/grain_vector_mechanisms/grain_cell_residual_v1"
    out.mkdir(exist_ok=False)
    files = {}
    for table in run.fn.TABLES:
        source = run.OUT / "analysis" / (table + ".csv")
        frame = pd.read_csv(source)
        if set(frame.cohort) != {"source4", "new8"} or any(
            c in frame for c in ("author_key", "text", "sentence", "user_id")
        ):
            raise ValueError("Unexpected public table schema")
        target = out / source.name
        target.write_bytes(source.read_bytes())
        files[source.name] = {"sha256": run.prev.old.sha(target), "rows": len(frame)}
    run.write(
        out / "publication.json",
        {
            "source_audit_sha256": run.prev.old.sha(run.OUT / "audit.json"),
            "source_manifest_sha256": run.prev.old.sha(run.OUT / "manifest.json"),
            "publisher_sha256": run.prev.old.sha(Path(__file__)),
            "scope": "Cohort-level cached native local-sensitivity decomposition, not prediction accuracy",
            "audit": audit,
            "files": files,
        },
    )
    print(
        {"status": "PUBLISHED_LOCAL_AGGREGATES", "tables": len(files), "path": str(out)}
    )


if __name__ == "__main__":
    publish()

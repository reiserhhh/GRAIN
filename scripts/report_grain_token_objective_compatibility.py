#!/usr/bin/env python3
"""Export NEXT61 source-level diagnostics without coefficient or author packets."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_token_objective_compatibility import OUT, read, sha, write


def publish():
    """Publish verified aggregate tables and an explicit diagnostic-only manifest."""
    audit = read(OUT / "replay_audit.json")
    summary = read(OUT / "primary/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Source audit required")
    dst = ROOT / "reports/grain_vector_mechanisms/token_objective_compatibility_v1"
    dst.mkdir(exist_ok=True)
    files = {}
    for name in ("source_objective_account.csv", "group_summary.csv"):
        source = OUT / "primary" / name
        if sha(source) != summary["files"][name]:
            raise ValueError("Source table differs")
        shutil.copyfile(source, dst / name)
        files[name] = sha(dst / name)
    write(
        dst / "publication.json",
        {
            "primary_commit": read(OUT / "manifest.json")["commit"],
            "audit": audit,
            "files": files,
            "unit": "Twenty source/query/reference function systems, not twenty independent author populations",
            "application_predictions": "Not computed for diagnostic coefficients",
        },
    )
    print({"public_tables": len(files), "directory": str(dst)})


if __name__ == "__main__":
    publish()

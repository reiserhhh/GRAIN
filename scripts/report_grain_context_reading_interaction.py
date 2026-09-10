#!/usr/bin/env python3
"""Publish only verified source/query aggregates for the NEXT63 intervention."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_context_reading_interaction import OUT, old


def publish():
    """Export three aggregate tables and a non-private provenance record."""
    audit = old.read(OUT / "audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Completed intervention audit required")
    dst = ROOT / "reports/grain_vector_mechanisms/context_reading_interaction_v1"
    dst.mkdir(exist_ok=True)
    files = {}
    for stem in ("context_summary", "action_summary", "factor_summary"):
        name = stem + ".csv"
        source = OUT / "analysis" / name
        if old.sha(source) != summary["files"][name]:
            raise ValueError("Frozen aggregate changed")
        if set(pd.read_csv(source, nrows=0).columns) & {
            "author_key",
            "text",
            "user_id",
            "tau",
            "changed_tau",
        }:
            raise ValueError("Individual material cannot be published")
        shutil.copyfile(source, dst / name)
        files[name] = old.sha(dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": old.read(OUT / "manifest.json")["commit"],
            "audit": audit,
            "files": files,
            "native_replay_checks": summary["replay_checks"],
            "unit": "Four source authors, two queries, four fixed GRAIN directions; not 32 independent people",
            "new_author_evaluation": False,
            "claim": "Finite H16 BODY-by-last reading dependence; not human psychological causality",
        },
    )
    print({"public_tables": len(files), "directory": str(dst)})


if __name__ == "__main__":
    publish()

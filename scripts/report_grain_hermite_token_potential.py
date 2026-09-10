#!/usr/bin/env python3
"""Publish NEXT62 cohort/query summaries, never author packets or fitted arrays."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_hermite_token_potential import OUT, old


def publish():
    """Export only the verified eight aggregate tables and their provenance."""
    audit = old.read(OUT / "replay_audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Completed same-function replay audit required")
    dst = ROOT / "reports/grain_vector_mechanisms/hermite_token_potential_v1"
    dst.mkdir(exist_ok=True)
    files = {}
    for stem in (
        "scores",
        "score_contrasts",
        "actions",
        "factor_actions",
        "action_contrasts",
        "factor_contrasts",
        "fields",
        "source_diagnostics",
    ):
        name = stem + ".csv"
        source = OUT / "analysis" / name
        if old.sha(source) != summary["files"][name]:
            raise ValueError("Aggregate source changed")
        columns = set(pd.read_csv(source, nrows=0).columns)
        if columns & {"author_key", "text", "body", "actual", "predicted", "user_id"}:
            raise ValueError("Individual-level material cannot be published")
        shutil.copyfile(source, dst / name)
        files[name] = old.sha(dst / name)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": old.read(OUT / "manifest.json")["commit"],
            "audit": audit,
            "criteria": summary["criteria"],
            "promotion": summary["promotion"],
            "files": files,
            "baseline": "Frozen NEXT60 kernel_then_mean; not refitted",
            "unit": "Source4 leave-one-out and exposed source-disjoint new8, reported separately",
            "targets": "Native model logit contrast and BODY actions, not human questionnaire scores",
            "encoder": "Cached original H16 remains a neural-encoder dependency",
        },
    )
    print({"public_tables": len(files), "directory": str(dst)})


if __name__ == "__main__":
    publish()

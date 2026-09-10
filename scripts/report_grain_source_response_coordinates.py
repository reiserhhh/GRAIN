#!/usr/bin/env python3
"""Publish NEXT75 aggregate evidence, including its runtime-only amendment."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_response_coordinates import OUT, TABLES, old, prev


def verify_amendment(out):
    """Require unchanged scientific stages and the declared compatibility code."""
    a = old.read(out / "analysis_runtime_amendment.json")
    files = {
        "wrapper_sha256": ROOT
        / "scripts/evaluate_grain_source_response_coordinates_compat.py",
        "test_sha256": ROOT / "tests/test_grain_response_coordinates_compat.py",
    }
    for field, path in files.items():
        if old.sha(path) != a[field]:
            raise ValueError("Runtime compatibility code changed")
    for stage, digest in a["frozen_stage_hashes"].items():
        prev.freeze(out, stage)
        if old.sha(out / stage / "freeze.json") != digest:
            raise ValueError("Pre-amendment scientific stage changed")
    if a["extra_fits"] != 0 or a["extra_predictions"] != 0:
        raise ValueError("Expected aggregation-only amendment")
    return a


def publish():
    """Copy exact public tables; average source-LOO diagnostics without identities."""
    manifest = old.verify(OUT)
    audit = old.read(OUT / "audit.json")
    summary = prev.freeze(OUT, "analysis")
    prev.freeze(OUT, "models")
    amendment = verify_amendment(OUT)
    if audit["status"] != "PASS" or set(summary["files"]) != set(TABLES):
        raise ValueError("Audited exact aggregate table inventory required")
    dst = ROOT / "reports/grain_vector_mechanisms/source_response_coordinates_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {
            "author_key",
            "user_id",
            "text",
            "token",
            "head",
            "reference",
        }:
            raise ValueError("Private fields in public aggregate")
        shutil.copyfile(src, dst / name)
    for name in ("source_capacity.csv", "source_training.csv"):
        frame = pd.read_csv(OUT / "models" / name).drop(columns="reference")
        frame.groupby(["source_authors", "query"], as_index=False).mean().to_csv(
            dst / name, index=False
        )
    basis = old.read(OUT / "models/basis_account.json")
    rows = []
    for reference, record in basis.items():
        for query, norms in record["normalizers"].items():
            rows.append(
                {
                    "source_authors": 4 if reference == "all4" else 3,
                    "query": query,
                    **{k: v for k, v in record.items() if k != "normalizers"},
                    **norms,
                }
            )
    pd.DataFrame(rows).groupby(
        ["source_authors", "query"], as_index=False
    ).mean().to_csv(dst / "source_basis.csv", index=False)
    names = (*TABLES, "source_capacity.csv", "source_training.csv", "source_basis.csv")
    old.write(
        dst / "publication.json",
        {
            "primary_commit": manifest["commit"],
            "audit": audit,
            "runtime_amendment": amendment,
            "cpu_scope": "8.173818 successful numeric stage CPU seconds including replay; failed aggregation CPU unrecorded; imports/preflight excluded",
            "files": {n: old.sha(dst / n) for n in names},
            "publisher_sha256": old.sha(Path(__file__)),
            "source_summary": "source_authors=3 averages four LOO fits; source_authors=4 is the single all-source reference",
            "scope": "Same-material scalar selector/key-channel responses, H16 and separate native V/fixed g retained; not human questionnaire accuracy",
        },
    )
    print({"status": "PUBLISHED", "tables": len(names)}, flush=True)


if __name__ == "__main__":
    publish()

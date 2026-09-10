#!/usr/bin/env python3
"""Publish finite geometry/operator aggregates with immutable replay evidence."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_material_geometry import verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def check(path, expected):
    """Reject stale numerical evidence rather than silently republishing it."""
    if sha(path) != expected:
        raise ValueError("Changed artifact: " + str(path))


def main():
    """Whitelist aggregate tables; no author rows, vectors or labels are copied."""
    out = BASE / "logic_material_geometry_v1_r2"
    replay = BASE / "logic_material_geometry_replay_v1"
    verify(out)
    verify(replay)
    f = read(out / "prediction_freeze.json")["files"]
    r = read(replay / "prediction_freeze.json")["files"]
    if f.keys() != r.keys() or len(f) != 2:
        raise ValueError("Two numerical outputs required")
    numeric = {}
    for name in f:
        check(out / name, f[name])
        check(replay / name, r[name])
        numeric[name] = f[name] == r[name]
    names = (
        "criterion_links.csv",
        "paired_contrasts.csv",
        "criterion_components.csv",
        "geometry_summary.csv",
    )
    summary = read(out / "analysis_r1/summary.json")
    aggregate = {}
    for name in names:
        check(out / "analysis_r1" / name, summary["files"][name])
        aggregate[name] = sha(out / "analysis_r1" / name) == sha(
            out / "analysis_replay" / name
        )
    operator = read(out / "operator_r1/summary.json")
    repeated_operator = read(out / "operator_replay/summary.json")
    op_replay = {}
    for folder in ("operator_r1", "operator_replay"):
        m = read(out / folder / "manifest.json")
        check(ROOT / "docs/GRAIN_MATERIAL_OPERATOR_EXTENSION_V1.md", m["plan_sha256"])
        check(ROOT / "scripts/derive_grain_material_operator.py", m["code_sha256"])
        check(
            ROOT / "grain_research/vector_mechanisms/material_operator.py",
            m["helper_sha256"],
        )
        check(out / "prediction_freeze.json", m["geometry_freeze_sha256"])
    for name, digest in operator["files"].items():
        check(out / "operator_r1" / name, digest)
        check(out / "operator_replay" / name, repeated_operator["files"][name])
        op_replay[name] = digest == repeated_operator["files"][name]
    diag = read(out / "fold_account_r1/summary.json")
    check(ROOT / "scripts/diagnose_grain_material_geometry.py", diag["code_sha256"])
    check(ROOT / "scripts/diagnose_grain_action_geometry.py", diag["helper_sha256"])
    check(
        ROOT / "docs/GRAIN_MATERIAL_FOLD_ACCOUNT_EXTENSION_V1.md", diag["plan_sha256"]
    )
    check(out / "fold_account_r1/fold_covariance.csv", diag["file_sha256"])
    aggregate["fold_covariance.csv"] = sha(
        out / "fold_account_r1/fold_covariance.csv"
    ) == sha(out / "fold_account_replay/fold_covariance.csv")
    if not all([*numeric.values(), *aggregate.values(), *op_replay.values()]):
        raise ValueError("Exact reproduction failed")
    files = {n: out / "analysis_r1" / n for n in names}
    files.update(
        {
            "source_references.csv": out / "source_references.csv",
            "fold_covariance.csv": out / "fold_account_r1/fold_covariance.csv",
            "summary.json": out / "analysis_r1/summary.json",
            "operator_summary.json": out / "operator_r1/summary.json",
            "fold_summary.json": out / "fold_account_r1/summary.json",
        }
    )
    dest = ROOT / "reports/grain_vector_mechanisms/material_geometry_tables_v1"
    dest.mkdir(exist_ok=True)
    forbidden = {"author_id", "user_id", "participant_id", "y_true", "text", "quote"}
    for name, source in files.items():
        if source.suffix == ".csv" and forbidden.intersection(
            pd.read_csv(source).columns
        ):
            raise ValueError("Private rows in aggregate")
        shutil.copyfile(source, dest / name)
    write(
        dest / "reproduction.json",
        {
            "numeric_byte_replay": numeric,
            "aggregate_byte_replay": aggregate,
            "operator_byte_replay": op_replay,
            "manifest_sha256": sha(out / "manifest.json"),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "plan_commit": "34a269b",
            "implementation_commit": "513cc56",
            "source_binding_fix_commit": "08703db",
            "operator_extension_commit": "0a56f36",
            "operator_commit": "f09fd2b",
            "fold_account_commit": "647fb5a",
            "initial_v1_preserved": "Stopped at mixed-cache duplicate check, before statistics or labels",
            "scoped_tests": 64,
            "new_tests": 14,
            "full_repository_suite": False,
            "private_rows_published": 0,
            "files": {n: sha(dest / n) for n in files},
        },
    )
    print({"aggregate_tables": 6, "exact_replays": 9, "private_rows_published": 0})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publish aggregate-only NEXT27 results after construction and analysis replay."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_material_coupling import verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Check frozen hashes and copy only the four declared aggregate tables."""
    out = BASE / "logic_material_coupling_v1_r2"
    replay = BASE / "logic_material_coupling_replay_v1"
    for folder in (out, replay):
        verify(folder)
    freeze = read(out / "prediction_freeze.json")["files"]
    again = read(replay / "prediction_freeze.json")["files"]
    if freeze.keys() != again.keys() or len(freeze) != 3:
        raise ValueError("Three matching numerical artifacts required")
    numerical = {}
    for name in freeze:
        if sha(out / name) != freeze[name] or sha(replay / name) != again[name]:
            raise ValueError("Numerical artifact changed")
        numerical[name] = freeze[name] == again[name]
    summary = read(out / "analysis_r1/summary.json")
    names = (
        "criterion_links.csv",
        "paired_association_deltas.csv",
        "centered_material_links.csv",
        "coverage.csv",
    )
    aggregate = {}
    for name in names:
        if sha(out / "analysis_r1" / name) != summary["files"][name]:
            raise ValueError("Analysis artifact changed")
        aggregate[name] = sha(out / "analysis_r1" / name) == sha(
            out / "analysis_replay" / name
        )
    if not all(numerical.values()) or not all(aggregate.values()):
        raise ValueError("Numerical reproduction failed")
    dest = ROOT / "reports/grain_vector_mechanisms/same_material_tables_v1"
    dest.mkdir(exist_ok=True)
    forbidden = {"author_id", "participant_id", "user_id", "y_true", "text", "quote"}
    for name in names:
        source = out / "analysis_r1" / name
        if forbidden.intersection(pd.read_csv(source).columns):
            raise ValueError("Private column in aggregate")
        shutil.copyfile(source, dest / name)
    shutil.copyfile(out / "analysis_r1/summary.json", dest / "summary.json")
    write(
        dest / "reproduction.json",
        {
            "numerical_byte_replay": numerical,
            "aggregate_byte_replay": aggregate,
            "manifest_sha256": sha(out / "manifest.json"),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "plan_commit": "85010ce",
            "implementation_commit": "e329bb4",
            "initial_v1_preserved": True,
            "revision_reason": "Tiny-signal fallback identity threshold; r2 numerical files identical to v1",
            "scoped_tests": 46,
            "full_repository_suite": False,
            "private_artifacts_published": False,
            "files": {name: sha(dest / name) for name in (*names, "summary.json")},
        },
    )
    print(
        {
            "aggregate_tables": len(names),
            "exact_replays": 7,
            "private_rows_published": 0,
        }
    )


if __name__ == "__main__":
    main()

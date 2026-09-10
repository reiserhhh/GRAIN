#!/usr/bin/env python3
"""Publish only the aggregate NEXT19/20 table whitelist after exact numerical replay."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_joint_carrier_readout import verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Reject author-level fields and AppleDouble files, retaining explicit source hashes."""
    out = BASE / "logic_joint_carrier_readout_v1"
    verify(out)
    predictions = read(out / "numerical_replay.json")
    if not all(predictions.values()) or len(predictions) != 3:
        raise ValueError("Three exact prediction/model replays required")
    names = ("joint_metrics.csv", "carrier_metrics.csv", "source_criterion_links.csv")
    checks = {
        n: sha(out / "analysis_r1" / n) == sha(out / "analysis_replay" / n)
        for n in names
    }
    if not all(checks.values()):
        raise ValueError("Aggregate replay differed")
    dest = ROOT / "reports/grain_vector_mechanisms/joint_carrier_readout_tables_v1"
    dest.mkdir(exist_ok=True)
    files = {n: out / "analysis_r1" / n for n in names}
    files["control_relative_accounts.csv"] = (
        out / "diagnostics/control_relative_accounts.csv"
    )
    for name, source in files.items():
        data = pd.read_csv(source)
        if any(
            k in data.columns
            for k in (
                "author_id",
                "participant_id",
                "y_true",
                "text",
                "candidate_logits",
            )
        ):
            raise ValueError("Private field in proposed public aggregate")
        shutil.copyfile(source, dest / name)
    for name, source in {
        "summary.json": out / "analysis_r1/summary.json",
        "diagnostic_summary.json": out / "diagnostics/diagnostic_summary.json",
    }.items():
        shutil.copyfile(source, dest / name)
    write(
        dest / "reproduction.json",
        {
            "prediction_replay": predictions,
            "aggregate_replay": checks,
            "manifest_sha256": sha(out / "manifest.json"),
            "plan_commit": "85f397f",
            "implementation_commit": "5b598bf",
            "post_result_diagnostic_commit": "8c511ed",
            "scoped_tests": 71,
            "new_tests": 5,
            "full_repository_suite": False,
            "files": {name: sha(dest / name) for name in files},
            "private_artifacts": "remain in ignored results, never copied by this whitelist",
        },
    )
    print(
        {
            "public_tables": len(files),
            "exact_numerical_replays": 6,
            "authors_published": 0,
        }
    )


if __name__ == "__main__":
    main()

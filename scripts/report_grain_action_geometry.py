#!/usr/bin/env python3
"""Publish the fixed aggregate whitelist for NEXT21-23 after exact cache replay."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_action_geometry import verify
from scripts.run_grain_joint_carrier_readout import check
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Verify numerical reproduction and copy aggregate-only, explicitly named files."""
    out = BASE / "logic_action_geometry_v1"
    replay = BASE / "logic_action_geometry_replay_v1"
    for source in (out, replay):
        verify(source)
    frozen = read(out / "prediction_freeze.json")["files"]
    replay_frozen = read(replay / "prediction_freeze.json")["files"]
    if frozen.keys() != replay_frozen.keys() or len(frozen) != 10:
        raise ValueError("Ten matching numerical artifacts required")
    numerical = {}
    for name, digest in frozen.items():
        check(out / name, digest)
        check(replay / name, replay_frozen[name])
        numerical[name] = digest == replay_frozen[name]
    names = (
        "carrier_metrics.csv",
        "joint_geometry_metrics.csv",
        "calibration_comparisons.csv",
        "direction_agreement.csv",
        "source_criterion_links.csv",
    )
    aggregate = {
        name: sha(out / "analysis_r1" / name) == sha(out / "analysis_replay" / name)
        for name in names
    }
    if not all(numerical.values()) or not all(aggregate.values()):
        raise ValueError("Exact numerical or aggregate replay differed")
    diagnostics = out / "condition_diagnostics_r2"
    d = read(diagnostics / "summary.json")
    check(ROOT / "scripts/diagnose_grain_action_geometry.py", d["code_sha256"])
    files = {name: out / "analysis_r1" / name for name in names}
    files.update(
        {
            "retained_condition_components.csv": diagnostics
            / "retained_condition_components.csv",
            "joint_fold_covariance.csv": diagnostics / "joint_fold_covariance.csv",
            "summary.json": out / "analysis_r1/summary.json",
            "condition_summary.json": diagnostics / "summary.json",
        }
    )
    dest = ROOT / "reports/grain_vector_mechanisms/action_geometry_tables_v1"
    dest.mkdir(exist_ok=True)
    forbidden = {
        "author_id",
        "participant_id",
        "y_true",
        "text",
        "candidate_logits",
        "token_ids",
        "hidden_states",
    }
    for name, source in files.items():
        if source.suffix == ".csv" and forbidden.intersection(
            pd.read_csv(source).columns
        ):
            raise ValueError("Private columns in proposed aggregate")
        shutil.copyfile(source, dest / name)
    write(
        dest / "reproduction.json",
        {
            "numerical_byte_replay": numerical,
            "aggregate_byte_replay": aggregate,
            "manifest_sha256": sha(out / "manifest.json"),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "plan_commit": "0fb89ff",
            "implementation_commit": "b30e767",
            "condition_diagnostic_commit": "8a45048",
            "condition_diagnostic_extension_sha256": d["code_sha256"],
            "scoped_tests": 83,
            "new_tests": 12,
            "full_repository_suite": False,
            "human_labels_for_fitting": False,
            "authors_published": 0,
            "files": {name: sha(dest / name) for name in files},
            "private_artifacts": "Remain in ignored results; no raw text, IDs or vectors copied",
        },
    )
    print({"aggregate_tables": 7, "exact_replays": 15, "authors_published": 0})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""No-fit post-result accounting for fold-specific compiled query coordinates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_grain_action_geometry import grouped_covariance
from scripts.run_grain_material_coupling import CRITERIA
from scripts.run_grain_material_geometry import ARMS, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def diagnose(out, name):
    """Preserve all query/factor/arm accounts; do not create a new predictor."""
    m = verify(out)
    if sha(CRITERIA) != m["criterion_sha256"]:
        raise ValueError("Changed source criteria")
    for n, expected in read(out / "prediction_freeze.json")["files"].items():
        if sha(out / n) != expected:
            raise ValueError("Changed frozen geometry")
    dest = out / name
    dest.mkdir(exist_ok=False)
    x = pd.read_csv(out / "private_geometry_views.csv", dtype={"author_id": str})
    x = (
        x.groupby(["author_id", "query", "factor"])
        .mean(numeric_only=True)
        .reset_index()
    )
    y = pd.read_csv(CRITERIA, dtype={"author_id": str}).set_index(
        ["author_id", "query"]
    )
    if y.index.duplicated().any():
        raise ValueError("Ambiguous criteria")
    rows = []
    for (q, factor), group in x.groupby(["query", "factor"], sort=True):
        group = group.sort_values("author_id")
        target = [y.loc[(uid, q), "y_true"] for uid in group.author_id]
        for arm in ARMS:
            rows.append(
                {
                    "query": q,
                    "factor": factor,
                    "arm": arm,
                    "n": len(group),
                    **grouped_covariance(group[arm], target, group.fold),
                }
            )
    table = pd.DataFrame(rows)
    if len(table) != 288 or table.closure_error.max() > 1e-12:
        raise ValueError("Incomplete covariance account")
    table.to_csv(dest / "fold_covariance.csv", index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "rows": len(table),
            "fits": 0,
            "new_neural_calls": 0,
            "code_sha256": sha(Path(__file__)),
            "helper_sha256": sha(ROOT / "scripts/diagnose_grain_action_geometry.py"),
            "plan_sha256": sha(
                ROOT / "docs/GRAIN_MATERIAL_FOLD_ACCOUNT_EXTENSION_V1.md"
            ),
            "closure_max": float(table.closure_error.max()),
            "file_sha256": sha(dest / "fold_covariance.csv"),
            "post_result_accounting": True,
            "p_values_added": False,
        },
    )
    print({"status": "COMPLETE", "accounts": len(table), "fits": 0})


def main():
    """Write a new immutable diagnostic directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_material_geometry_v1_r2"
    )
    parser.add_argument("--name", default="fold_account_r1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Use ignored results and a simple name")
    diagnose(out, args.name)


if __name__ == "__main__":
    main()

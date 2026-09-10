#!/usr/bin/env python3
"""Account for why nested energy projections need not improve a measured action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_position_reader import verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def addition_account(y, before, after):
    """Exact ordered squared-error change, not an independent variance allocation."""
    y, a, b = (np.asarray(x, float) for x in (y, before, after))
    if (
        y.ndim != 1
        or y.shape != a.shape
        or y.shape != b.shape
        or not np.isfinite([y, a, b]).all()
        or np.sum(y * y) == 0
    ):
        raise ValueError("Finite aligned nonzero target actions required")
    norm = np.sum(y * y)
    old = a - y
    increment = b - a
    result = {
        "old_action_error": float(np.sum(old**2) / norm),
        "new_action_error": float(np.sum((b - y) ** 2) / norm),
        "added_action_energy": float(np.sum(increment**2) / norm),
        "prior_added_cross": float(2 * np.sum(old * increment) / norm),
    }
    result["identity_error"] = (
        result["new_action_error"]
        - result["old_action_error"]
        - result["added_action_energy"]
        - result["prior_added_cross"]
    )
    if abs(result["identity_error"]) > 1e-12:
        raise ValueError("Mode-addition identity failed")
    return result


def analyze(out, name):
    """Read frozen field/contraction tables, adding no model or selected rank."""
    m = verified(out)
    f = read(out / "prediction_freeze.json")
    if sha(out / "manifest.json") != f["manifest_sha256"]:
        raise ValueError("Construction manifest changed")
    for p, d in f["files"].items():
        if sha(out / p) != d:
            raise ValueError("Frozen prediction changed")
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "new_fits": 0,
            "human_labels_read": False,
            "new_neural_calls": 0,
        },
    )
    rows = []
    for q in m["queries"]:
        actions = pd.read_csv(out / q / "private_actions.csv")
        fields = pd.read_csv(out / q / "private_fields.csv")
        for method in ("local_svd", "shared_svd"):
            for old, new in ((1, 2), (2, 4), (4, 8)):
                a = actions[
                    (actions.method == method) & (actions["rank"] == old)
                ].sort_values(["index", "factor"])
                b = actions[
                    (actions.method == method) & (actions["rank"] == new)
                ].sort_values(["index", "factor"])
                if not np.array_equal(
                    a[["index", "factor", "actual"]].to_numpy(),
                    b[["index", "factor", "actual"]].to_numpy(),
                ):
                    raise ValueError("Action pairing changed")
                fa = fields[
                    (fields.method == method) & (fields["rank"] == old)
                ].sort_values("index")
                fb = fields[
                    (fields.method == method) & (fields["rank"] == new)
                ].sort_values("index")
                if np.any(
                    fb.field_error_energy.to_numpy()
                    > fa.field_error_energy.to_numpy() + 1e-12
                ):
                    raise ValueError("Nested projection increased field error")
                rows.append(
                    {
                        "query": q,
                        "method": method,
                        "rank_from": old,
                        "rank_to": new,
                        "authors": 4,
                        "gradient_error_change_mean": float(
                            fb.field_error_ratio.mean() - fa.field_error_ratio.mean()
                        ),
                        **addition_account(
                            a.actual.to_numpy(),
                            a.predicted.to_numpy(),
                            b.predicted.to_numpy(),
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(dest / "mode_addition.csv", index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "new_fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "identity_error_max": max(abs(r["identity_error"]) for r in rows),
            "field_improvement_action_worsening_cases": sum(
                r["new_action_error"] > r["old_action_error"]
                and r["gradient_error_change_mean"] < 0
                for r in rows
            ),
            "comparisons": len(rows),
            "files": {"mode_addition.csv": sha(dest / "mode_addition.csv")},
        },
    )
    print({"status": "MODE_ACCOUNT_COMPLETE", "comparisons": len(rows)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_position_reader_v1")
    p.add_argument("--name", default="mode_analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.name)

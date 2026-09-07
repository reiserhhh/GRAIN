#!/usr/bin/env python3
"""S12 paired target recovery contrasts, without refitting or new label access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_real_author_path import BASE, read, sha, write


def mse_gain(target, prediction, control, null, draws=1000):
    """Report paired MSE reduction, normalized by the out-of-fold mean-control error."""
    y, p, c, n = [np.asarray(x, float) for x in (target, prediction, control, null)]
    if y.ndim != 1 or any(a.shape != y.shape for a in (p, c, n)):
        raise ValueError("Require paired vectors")
    if not np.isfinite([y, p, c, n]).all():
        raise ValueError("Finite target comparison required")
    gain, base = (c - y) ** 2 - (p - y) ** 2, (n - y) ** 2
    if not base.mean() > 0:
        raise ValueError("Nonzero null error required")
    ix = np.random.default_rng(732).integers(len(y), size=(draws, len(y)))
    estimates = gain[ix].mean(1) / base[ix].mean(1)
    return {
        "mse_gain_over_null": float(gain.mean() / base.mean()),
        "mse_gain_lower": float(np.quantile(estimates, 0.025)),
        "mse_gain_upper": float(np.quantile(estimates, 0.975)),
    }


def main():
    """Compare every frozen representation to matched body and earlier E5 readouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    out = args.analysis.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private analysis stays in ignored results")
    path = out / "private_oof_before_outcomes.csv"
    if sha(path) != read(out / "summary.json")["oof_sha256"]:
        raise ValueError("S12 OOF changed")
    prior = BASE / "contrast_student_s11_r1"
    old_path = prior / "private_oof_before_outcomes.csv"
    if sha(old_path) != read(prior / "summary.json")["oof_sha256"]:
        raise ValueError("S11 OOF changed")
    frame = pd.read_csv(path, dtype={"author_id": str})
    old = pd.read_csv(old_path, dtype={"author_id": str})
    rows = []
    for (rep, method, target), g in frame.groupby(
        ["representation", "method", "target"]
    ):
        g = g.set_index("author_id")
        body = (
            frame[
                (frame.representation == "body_mean")
                & (frame.method == method)
                & (frame.target == target)
            ]
            .set_index("author_id")
            .loc[g.index]
        )
        e5 = (
            old[(old.method == method) & (old.target == target)]
            .set_index("author_id")
            .loc[g.index]
        )
        for name, control in (("body_mean", body), ("s11_e5", e5)):
            if not np.allclose(g.teacher, control.teacher, rtol=1e-12, atol=1e-12):
                raise ValueError("Teacher targets mismatch")
            rows.append(
                {
                    "representation": rep,
                    "method": method,
                    "target": target,
                    "control": name,
                    **paired_intervals(
                        g.teacher, g.prediction, control.prediction, False
                    ),
                    **mse_gain(
                        g.teacher, g.prediction, control.prediction, g.null_prediction
                    ),
                }
            )
    destination = out / "target_recovery"
    destination.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(destination / "paired_target_intervals.csv", index=False)
    write(
        destination / "summary.json",
        {
            "status": "COMPLETE",
            "script_sha256": sha(__file__),
            "oof_sha256": sha(path),
            "new_fits": 0,
            "labels_read": False,
            "scope": "pointwise paired authors conditional on frozen OOF predictions",
            "rows": rows,
        },
    )
    selected = pd.DataFrame(rows)
    print(selected[selected.target == "configuration_response"].to_string(index=False))


if __name__ == "__main__":
    main()

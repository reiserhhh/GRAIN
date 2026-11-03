#!/usr/bin/env python3
"""Retain total and source-relative criterion components;no score recalibration."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_conditional_publication as C
from scripts import run_grain_conditional_publication_hm292 as R


def covariance(x, y):
    """Compute a descriptive paired covariance in unchanged score units."""
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def main():
    """Describe the actual meaning of a within-profile calibrated SOURCE ruler."""
    parent = R.DEFAULT
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    out = parent / "postoutcome_criterion_r1"
    out.mkdir(exist_ok=False)
    R.R.write(
        out / "manifest.json",
        {
            "classification": "POSTOUTCOME criterion interpretation only,no field/ruler/primary change",
            "code": R.R.digest(Path(__file__)),
            "inputs": {
                p: R.R.digest(parent / p)
                for p in (
                    "observations.npz",
                    "measured_fields.npz",
                    "criterion_contacts.csv",
                )
            },
            "no_new_fit_permutation_or_held_labels": True,
        },
    )
    z, f = (
        R.R.load(parent / "observations.npz"),
        R.R.load(parent / "measured_fields.npz"),
    )
    rows, errors = [], []
    for bank in (0, 1):
        cal = z["source_labels"][0, z["bank"] == bank]
        y = z["source_labels"][0, z["bank"] != bank]
        for q, trait in enumerate(R.TRAITS):
            cal_groups = np.delete(cal, q, 1).astype(int) @ (1 << np.arange(4))
            test_groups = np.delete(y, q, 1).astype(int) @ (1 << np.arange(4))
            supported = np.isin(test_groups, np.unique(cal_groups))
            m = np.full(len(y), np.nan)
            for group in np.unique(cal_groups):
                m[test_groups == group] = cal[cal_groups == group, q].mean()
            e = y[:, q] - m
            for part in C.PARTS:
                for view in (0, 1):
                    for assignment in (0, 1):
                        x = f[f"bank{bank}/{part}"][:, view, assignment, q]
                        xs = x[supported]
                        total, composition, within = (
                            covariance(xs, y[supported, q]),
                            covariance(xs, m[supported]),
                            covariance(xs, e[supported]),
                        )
                        errors.append(abs(total - composition - within))
                        rows.append(
                            {
                                "calibration_bank": bank,
                                "view": view,
                                "calibration_assignment": ("actual", "old_permuted")[
                                    assignment
                                ],
                                "part": part,
                                "trait": trait,
                                "n_all": len(y),
                                "n_conditional_support": int(supported.sum()),
                                "total_cov": total,
                                "composition_cov": composition,
                                "within_cov": within,
                                "r_total": C.correlation(xs, y[supported, q]),
                                "r_source_relative_within": C.correlation(
                                    xs, e[supported]
                                ),
                            }
                        )
    with (out / "criterion_components.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "missing_conditional_authors": sorted(
            {r["n_all"] - r["n_conditional_support"] for r in rows}
        ),
        "covariance_closure_error": max(errors),
        "not_causal_trait_isolation": True,
        "primary_unchanged": True,
    }
    if max(errors) > 1e-10:
        raise ArithmeticError("Criterion components changed the observed contact")
    R.R.write(out / "summary.json", summary)
    R.R.seal(
        out, "seal.json", ["manifest.json", "criterion_components.csv", "summary.json"]
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

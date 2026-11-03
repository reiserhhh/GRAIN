#!/usr/bin/env python3
"""Read-only POSTOUTCOME alignment/amplitude account; no scientific rescue or fit."""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_behavior_pushforward import FEATURES
from scripts import run_grain_behavior_pushforward_hm291 as R


def account(y, prediction, prior, variance, donors):
    """Decompose fixed prediction gain; donor bijection preserves mean and variance."""
    b, d = y - prior, prediction - prior
    covariance = ((b - b.mean(0)) * (d - d.mean(0))).mean(0)
    fields = np.array(
        [
            2 * covariance / variance,
            2 * b.mean(0) * d.mean(0) / variance,
            -d.var(0) / variance,
            -(d.mean(0) ** 2) / variance,
        ]
    )
    literal = (b**2 - (y - prediction) ** 2).mean(0) / variance
    stranger = (
        2 * ((y - y.mean(0)) * (prediction - prediction[donors])).mean(0) / variance
    )
    literal_stranger = ((y - prediction[donors]) ** 2 - (y - prediction) ** 2).mean(
        0
    ) / variance
    return (
        fields,
        literal,
        stranger,
        max(
            float(np.max(abs(fields.sum(0) - literal))),
            float(np.max(abs(stranger - literal_stranger))),
        ),
    )


def main():
    """Append a separate sealed descriptive account to immutable primary outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    m = R.check(args.parent, ("computed_seal.json", "verification_seal.json"))
    out = args.parent / "postoutcome_account_r1"
    out.mkdir(exist_ok=False)
    code = "scripts/describe_grain_behavior_pushforward_hm291.py"
    inputs = ("observations.npz", "predictions.npz", "summary.json")
    R.write(
        out / "manifest.json",
        {
            "classification": "POSTOUTCOME descriptive explanation,not a new primary or fitting experiment",
            "code": {code: R.digest(ROOT / code)},
            "inputs": {p: R.digest(args.parent / p) for p in inputs},
            "no_fit_new_draws_model_calls": True,
        },
    )
    z, p = (
        R.load(args.parent / "observations.npz"),
        R.load(args.parent / "predictions.npz"),
    )
    start, rows, errors = time.monotonic(), [], []
    pooled = {}
    for ref in m["config"]["references"]:
        for model in R.MODELS:
            f, gain, stranger, error = account(
                z["actual_later_behavior"],
                p[f"{ref}_{model}"],
                p[f"{ref}_prior"],
                p[f"{ref}_variance"],
                z["donors"],
            )
            errors.append(error)
            for j, feature in enumerate(FEATURES):
                rows.append(
                    {
                        "reference": ref,
                        "model": model,
                        "feature": feature,
                        "alignment": float(f[0, j]),
                        "reference_shift": float(f[1, j]),
                        "spread_cost": float(f[2, j]),
                        "mean_shift_cost": float(f[3, j]),
                        "gain": float(gain[j]),
                        "own_stranger_alignment": float(stranger[j]),
                    }
                )
            if ref == "pooled":
                pooled[model] = {
                    k: float(v)
                    for k, v in zip(
                        (
                            "alignment",
                            "reference_shift",
                            "spread_cost",
                            "mean_shift_cost",
                        ),
                        f.mean(1),
                        strict=True,
                    )
                }
                pooled[model]["gain"] = float(gain.mean())
    with (out / "risk_account.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "pooled": pooled,
        "literal_error": max(errors),
        "seconds": time.monotonic() - start,
        "source_and_scientific_results_unchanged": True,
        "not_a_causal_error_fraction": True,
    }
    R.write(out / "summary.json", result)
    R.seal(out, "seal.json", ["manifest.json", "risk_account.csv", "summary.json"])
    if max(errors) > 1e-10:
        raise ArithmeticError("Descriptive account does not match the fixed risk")
    print(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recover PV01 numeric effects and reclassify unsupported precision labels."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    """Audit fixed native effects without claiming unpersisted gradients exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = list(csv.DictReader((args.input / "application_effects.csv").open()))
    checks = json.loads((args.input / "gradient_checks.json").read_text())
    triage = []
    for case in checks:
        for h in (0.01, 0.005):
            key = f"h_{h:g}"
            actual = abs(case[f"fd_{key}"] - case["analytic_directional_derivative"])
            np.testing.assert_allclose(actual, case[f"abs_error_{key}"], atol=1e-15)
            if actual <= case[f"tolerance_{key}"]:
                raise ValueError("legacy six-failure assumption changed")
        triage.append(
            {
                "pair": case["pair"],
                "query": case["query"],
                "old_status": "PRECISION_LIMITED",
                "current_cause": "UNKNOWN",
                "error_h01": case["abs_error_h_0.01"],
                "error_h005": case["abs_error_h_0.005"],
                "halving_error_ratio": case["abs_error_h_0.005"]
                / case["abs_error_h_0.01"],
                "reason": "two FP32/MPS scales do not isolate formula, precision or curvature; raw plus/minus and gradient arrays absent",
            }
        )
    rows = []
    for query in ("extraversion", "neuroticism"):
        subset = [r for r in raw if r["query"] == query]
        if len(subset) != 72 or len({r["pair"] for r in subset}) != 72:
            raise ValueError("PV01 pair coverage mismatch")
        for target, field in (
            ("full_text_delta", "contrast"),
            ("native_donor", "donor_effect"),
        ):
            y = np.array([float(r[field]) for r in subset])
            for method in ("zero", "euclidean", "weighted"):
                p = (
                    np.zeros_like(y)
                    if method == "zero"
                    else np.array([float(r[method + "_effect"]) for r in subset])
                )
                rows.append(
                    {
                        "query": query,
                        "target": target,
                        "method": method,
                        "n": len(y),
                        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                        "mse_delta_zero": float(np.mean((p - y) ** 2 - y**2)),
                        "pearson": None
                        if method == "zero"
                        else float(np.corrcoef(p, y)[0, 1]),
                    }
                )
    legacy = list(csv.DictReader((args.input / "metrics.csv").open()))
    max_error = 0.0
    for row in rows:
        if row["method"] == "zero":
            continue
        old = next(
            r
            for r in legacy
            if all(r[k] == row[k] for k in ("query", "method", "target"))
        )
        max_error = max(
            max_error,
            abs(float(old["rmse"]) - row["rmse"]),
            abs(float(old["r"]) - row["pearson"]),
        )
    if max_error > 1e-10:
        raise ValueError("PV01 summary replay failed")
    for name, values in (("metrics.csv", rows), ("six_failure_triage.csv", triage)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = {
        "status": "CACHE_REPLAY_PASS_GRADIENT_CAUSES_UNKNOWN",
        "application_pairs_per_query": 72,
        "metrics_replay_max_error": max_error,
        "failures": len(triage),
        "new_forward_calls": 0,
        "native_common_chart": False,
        "source_refit": "missing persisted source gradients and fitted weight arrays",
        "hashes": {
            name: hashlib.sha256((args.input / name).read_bytes()).hexdigest()
            for name in (
                "application_effects.csv",
                "gradient_checks.json",
                "metrics.csv",
                "manifest.json",
            )
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

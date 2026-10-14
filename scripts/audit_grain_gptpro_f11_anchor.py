#!/usr/bin/env python3
"""Alternate scalar covariance and source-frozen score audit for anchor worlds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    """Rebuild all application scores and report paired fixed-fit uncertainty."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.input / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    cells = [(k, r, t) for k in range(3) for r in range(3) for t in range(2)]
    max_error = 0.0
    rebuilt = {}
    for path in args.input.glob("fit_*.json"):
        seed, world, method = path.stem.removeprefix("fit_").split("_", 2)
        model = json.loads(path.read_text())
        s = np.array(model["sigma"])
        v = np.array(
            [
                [
                    s[k, kk]
                    + model["method"] * (r == rr)
                    + model["error"] * (i == j)
                    + model.get("actor", 0.0)
                    for j, (kk, rr, tt) in enumerate(cells)
                ]
                for i, (k, r, t) in enumerate(cells)
            ]
        )
        cross = np.column_stack([s[:, k] for k, r, t in cells])
        inv = np.linalg.inv(v)
        variance = s - cross @ inv @ cross.T
        for role in ("validation", "application"):
            with np.load(args.input / f"observations_{seed}_{role}.npz") as z:
                y = z["text"]
            p = (y - np.array(model["mean"])) @ inv @ cross.T
            rebuilt[seed, world, method, role] = p, variance
    for row in rows:
        p, v = rebuilt[row["seed"], row["world"], row["method"], row["role"]]
        i, k = int(row["index"]), int(row["trait"])
        max_error = max(
            max_error,
            abs(p[i, k] - float(row["prediction"])),
            abs(v[k, k] - float(row["variance"])),
        )
    with (args.input / "metrics.csv").open() as stream:
        metric_rows = list(csv.DictReader(stream))
    for row in metric_rows:
        p, v = rebuilt[row["seed"], row["world"], row["method"], row["role"]]
        with np.load(args.input / f"truth_{row['seed']}_{row['role']}.npz") as z:
            truth = z["f" if row["world"] == "A" else "g"]
        mse = np.mean((p - truth) ** 2)
        coverage = np.mean(np.abs(p - truth) <= 1.96 * np.sqrt(np.diag(v)))
        max_error = max(
            max_error,
            abs(mse - float(row["mse"])),
            abs(coverage - float(row["coverage"])),
        )
    intervals = []
    seeds = sorted({r["seed"] for r in rows})
    for world in ("A", "B"):
        differences = []
        for seed in seeds:
            with np.load(args.input / f"truth_{seed}_application.npz") as z:
                truth = z["f" if world == "A" else "g"]
            anchored = rebuilt[seed, world, "source_anchored", "application"][0]
            naive = rebuilt[seed, world, "unanchored_effective", "application"][0]
            differences.append(
                np.mean((anchored - truth) ** 2 - (naive - truth) ** 2, axis=1)
            )
        rng = np.random.default_rng(20260915)
        boot = np.array(
            [
                np.mean(
                    np.concatenate(
                        [rng.choice(d, len(d), replace=True) for d in differences]
                    )
                )
                for _ in range(2000)
            ]
        )
        low, high = np.quantile(boot, [0.025, 0.975])
        intervals.append(
            {
                "world": world,
                "mse_delta": float(np.mean(differences)),
                "ci_low": float(low),
                "ci_high": float(high),
                "scope": "fixed source fits; application-author bootstrap stratified by seed",
            }
        )
    if max_error > 1e-10:
        raise ValueError("anchor numerical replay failed")
    result = {
        "status": "PASS_SCORE_AND_METRIC_REPLAY",
        "prediction_rows": len(rows),
        "metric_rows": len(metric_rows),
        "max_error": max_error,
        "paired_intervals": intervals,
        "source_parameter_audit": "exact population recovery unit test; finite source fit independent reconstruction pending",
        "independent_agent": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

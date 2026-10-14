#!/usr/bin/env python3
"""Rebuild every supported score and aggregate transport table from saved inputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def main():
    """Verify transformed observations, support, calibrated values and all metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.input / "protocol_before_run.json").read_text())
    with (args.input / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    maximum = 0.0
    scalar_count = 0
    null_count = 0
    for row in rows:
        seed, world, role, method = [
            row[k] for k in ("seed", "world", "role", "method")
        ]
        source_role = "pilot" if role == "pilot_validation" else "application"
        with np.load(args.input / f"observations_{seed}_{source_role}.npz") as z:
            old = z["old"]
            condition = z["conditions"]
        with np.load(args.input / f"truth_{seed}_{source_role}.npz") as z:
            theta = z["theta"]
        if role == "pilot_validation":
            old, theta = old[config["pilot_fit"] :], theta[config["pilot_fit"] :]
        if world == "identity":
            new = old.copy()
        elif world == "global":
            new = 1 + 1.5 * old
        elif world == "conditional":
            new = condition + (1 + 0.25 * condition) * old
        else:
            new = old + 0.1 * old**3
        with np.load(args.input / f"new_{seed}_{world}_{role}.npz") as z:
            maximum = max(maximum, float(np.max(np.abs(z["new"] - new))))
        if method == "uncalibrated":
            prediction = new
            support = np.ones_like(new, dtype=bool)
        else:
            model = json.loads(
                (args.input / f"calibration_{seed}_{world}_{method}.json").read_text()
            )
            prediction = np.full(new.shape, np.nan)
            support = np.zeros_like(new, dtype=bool)
            if model["status"] == "READY":
                c = np.array(model["coefficients"])
                support = np.broadcast_to(
                    np.isin(condition, model["conditions"]), new.shape
                ).copy()
                if method == "cubic":
                    support &= (new >= model["new_range"][0]) & (
                        new <= model["new_range"][1]
                    )
                    values = c[0] + new * (c[1] + c[2] * new * new)
                elif method == "global":
                    values = (new - c[0]) / c[1]
                else:
                    scale = c[2] + c[3] * condition
                    support &= np.abs(scale) > 1e-12
                    values = (new - c[0] - c[1] * condition) / scale
                prediction[support] = values[support]
        with np.load(args.input / f"scores_{seed}_{world}_{role}_{method}.npz") as z:
            np.testing.assert_array_equal(z["support"], support)
            np.testing.assert_array_equal(np.isnan(z["score"]), ~support)
            if support.any():
                maximum = max(
                    maximum,
                    float(np.max(np.abs(z["score"][support] - prediction[support]))),
                )
        scalar_count += prediction.size
        null_count += int((~support).sum())
        valid = np.all(support[:, :24], axis=1)
        assert int(row["supported_authors"]) == int(valid.sum())
        assert int(row["supported_readings"]) == int(support.sum())
        if not valid.any():
            assert all(
                row[k] == ""
                for k in ("cell_mse", "delta_mse", "delta_spearman", "delta_coverage")
            )
            continue
        z = condition[:24]
        intercept = prediction[valid, :24].mean(axis=1)
        slope = prediction[valid, :24] @ z / (z @ z)
        target = theta[valid]
        delta = 2 * slope
        truth = 2 * target[:, 1]
        source = json.loads((args.input / f"source_reference_{seed}.json").read_text())
        variance = 4 * source["estimated_error_variance"] / (z @ z)
        r1, r2 = rankdata(delta), rankdata(truth)
        r1 -= r1.mean()
        r2 -= r2.mean()
        values = {
            "material_mse": np.mean((prediction[support] - old[support]) ** 2),
            "cell_mse": np.mean(
                (
                    (intercept - target[:, 0])[:, None]
                    + (slope - target[:, 1])[:, None] * condition[:4]
                )
                ** 2
            ),
            "delta_mse": np.mean((delta - truth) ** 2),
            "delta_coverage": np.mean(
                np.abs(delta - truth) <= 1.96 * np.sqrt(variance)
            ),
            "delta_spearman": np.dot(r1, r2) / np.sqrt(np.dot(r1, r1) * np.dot(r2, r2)),
        }
        maximum = max(
            maximum, max(abs(value - float(row[k])) for k, value in values.items())
        )
    if maximum > 1e-9:
        raise ValueError("full transport table audit failed")
    summary = {
        "status": "PASS_FULL_SCORE_SUPPORT_AND_AGGREGATE_REPLAY",
        "score_scalars": scalar_count,
        "missing_scalars": null_count,
        "aggregate_rows": len(rows),
        "max_difference": maximum,
        "independent_agent": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

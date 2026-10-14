#!/usr/bin/env python3
"""Recalculate the final dynamic tables without fitting or changing predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


def main():
    """Check all final target rows, covariance sums and coverage decisions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_csv(args.input / "predictions.csv")
    metrics = pd.read_csv(args.input / "metrics.csv")
    keys = ["world", "seed", "author_id", "budget", "target"]
    assert not predictions.duplicated(keys).any()
    indexed = predictions.set_index(keys)
    targets = ["current_state", "future_observation_k4", "future_observation_average4"]
    count = 0
    for world in ("core", "near_root", "shared_shock"):
        for seed in (20260915, 20260916, 20260917):
            records = json.loads(
                (args.input / f"covariances_{world}_{seed}.json").read_text()
            )
            assert len(records) == 600
            for record in records:
                total = sum(
                    np.asarray(record[k])
                    for k in ("event_state", "personal_parameter", "source")
                )
                np.testing.assert_allclose(total, record["total"], atol=1e-12)
                for i, target in enumerate(targets):
                    row = indexed.loc[
                        (world, seed, record["author_id"], record["budget"], target)
                    ]
                    np.testing.assert_allclose(
                        [row["mean"], row["variance"]],
                        [record["mean"][i], total[i, i]],
                        atol=1e-12,
                    )
                    count += 1
    metric_keys = ["world", "seed", "budget", "target"]
    assert not metrics.duplicated(metric_keys).any()
    metric_index = metrics.set_index(metric_keys)
    for key, group in predictions.groupby(metric_keys):
        error = group["mean"] - group["truth"]
        covered = abs(error) <= 1.96 * np.sqrt(group["variance"])
        low, high = proportion_confint(int(covered.sum()), len(group), method="wilson")
        expected = [
            len(group),
            (error**2).mean(),
            covered.mean(),
            (3.92 * np.sqrt(group["variance"])).mean(),
            low,
            high,
        ]
        row = metric_index.loc[key]
        np.testing.assert_allclose(
            row[
                ["n", "mse", "coverage", "mean_width", "wilson_low", "wilson_high"]
            ].to_numpy(dtype=float),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )
        decision = (
            "DESCRIPTIVE_DEPENDENT_AUTHORS"
            if key[0] == "shared_shock"
            else "PASS"
            if low <= 0.95 <= high
            else "FAIL"
        )
        assert row["decision"] == decision
    assert count == len(predictions) == 16200 and len(metrics) == 81
    result = {
        "status": "FINAL_TABLE_REPLAY_PASS",
        "prediction_rows": count,
        "metric_rows": len(metrics),
        "scope": "covariance-to-table and metric replay; not a new calibration experiment; Wilson uses statsmodels normal quantile versus production rounded 1.96",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

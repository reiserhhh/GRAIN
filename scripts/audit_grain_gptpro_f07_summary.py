#!/usr/bin/env python3
"""Alternate-formula covariance, aggregate and paired-interval replay for F07."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from audit_grain_gptpro_f07_estimation import probabilities


def main():
    """Rebuild uncertainty using numerical probability derivatives and count weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    checkpoint = json.loads((args.input / "checkpoint.json").read_text())
    if checkpoint["status"] != "COMPUTED_AUDIT_PENDING":
        raise ValueError("complete computation required")
    application = {}
    covariance_max = 0.0
    covariance_count = 0
    for path in sorted(args.input.glob("author_*.json.gz")):
        with gzip.open(path, "rt") as stream:
            packet = json.load(stream)
        for row in packet["rows"]:
            theta = np.array(row["estimate"])
            precision = np.diag([1.0, 4.0])
            for counts, z in zip(row["counts"], (-1, 1), strict=True):
                p = probabilities(theta, z, row)
                h = 1e-5
                derivative = np.column_stack(
                    [
                        (
                            probabilities(theta + h * unit, z, row)
                            - probabilities(theta - h * unit, z, row)
                        )
                        / (2 * h)
                        for unit in np.eye(2)
                    ]
                )
                for probability, gradient in zip(p, derivative, strict=True):
                    if probability > 0:
                        precision += (
                            sum(counts) * np.outer(gradient, gradient) / probability
                        )
            # Explicit 2x2 inverse independent of the fitting implementation.
            a, b, d = precision[0, 0], precision[0, 1], precision[1, 1]
            covariance = np.array([[d, -b], [-b, a]]) / (a * d - b * b)
            saved = np.asarray(row["covariance"])
            covariance_max = max(
                covariance_max, float(np.max(np.abs(covariance - saved)))
            )
            np.testing.assert_allclose(covariance, saved, atol=1e-8, rtol=1e-7)
            covariance_count += 1
        if packet["role"] == "application":
            application.setdefault(packet["seed"], []).append(packet)
    if covariance_count != checkpoint["fits"]:
        raise ValueError("fit count mismatch")
    seeds = manifest["seeds"]
    count = manifest["roles_per_seed"]["application"]
    if any(len(application[s]) != count for s in seeds):
        raise ValueError("application count mismatch")
    errors = np.array(
        [
            [[row["error"] for row in packet["rows"]] for packet in application[s]]
            for s in seeds
        ]
    )
    squared = errors**2
    designs = manifest["designs"]
    lookup = {
        (
            r["world"],
            r["variant"],
            str(r["level"]) if r["level"] is not None else "",
            r["budget"],
        ): i
        for i, r in enumerate(designs)
    }
    maximum_metric_difference = 0.0
    metric_count = 0
    for filename, by_seed in (
        ("application_metrics.csv", False),
        ("metrics_by_seed.csv", True),
    ):
        for row in csv.DictReader((args.summary / filename).open()):
            index = lookup[
                tuple(row[k] for k in ("world", "variant", "level", "budget"))
            ]
            target = ("a", "b").index(row["target"])
            values = (
                squared[seeds.index(int(row["seed"])), :, index, target]
                if by_seed
                else squared[:, :, index, target]
            )
            difference = abs(float(values.mean()) - float(row["mse"]))
            maximum_metric_difference = max(maximum_metric_difference, difference)
            np.testing.assert_allclose(values.mean(), float(row["mse"]), atol=1e-12)
            metric_count += 1
    # Same declared resamples, evaluated via multiplicity weights rather than slicing.
    rng = np.random.default_rng(20260915)
    weights = []
    for _ in seeds:
        draws = rng.integers(0, count, size=(2000, count))
        weights.append(
            np.array([np.bincount(x, minlength=count) for x in draws]) / count
        )
    intervals = 0
    interval_max = 0.0
    for row in csv.DictReader((args.summary / "paired_level1_comparisons.csv").open()):
        key = tuple(row[k] for k in ("world", "variant", "level", "budget"))
        index = lookup[key]
        reference = lookup[(key[0], key[1], "1", key[3])]
        target = ("a", "b").index(row["target"])
        delta = squared[:, :, index, target] - squared[:, :, reference, target]
        boot = sum(w @ d for w, d in zip(weights, delta, strict=True)) / len(seeds)
        replay = [delta.mean(), *np.quantile(boot, [0.025, 0.975])]
        expected = [float(row[k]) for k in ("mse_delta", "ci_low", "ci_high")]
        interval_max = max(
            interval_max, float(np.max(np.abs(np.array(replay) - expected)))
        )
        np.testing.assert_allclose(replay, expected, atol=1e-12)
        intervals += 1
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "COVARIANCE_MSE_PAIRED_INTERVAL_REPLAY_PASS",
        "covariance_rows": covariance_count,
        "max_covariance_difference": covariance_max,
        "mse_rows": metric_count,
        "max_mse_difference": maximum_metric_difference,
        "paired_intervals": intervals,
        "max_interval_difference": interval_max,
        "scope": "same-agent alternate formulas; covariance numerical-derivative tolerance1e-8/1e-7; no new fits",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

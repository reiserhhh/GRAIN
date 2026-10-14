#!/usr/bin/env python3
"""Summarize all F07 application arms, preserving paired author-level uncertainty."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def save_csv(path, rows):
    """Write a generated table with explicit headers."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Bootstrap whole authors within each seed, never individual opportunities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads((args.input / "checkpoint.json").read_text())
    if checkpoint["status"] != "COMPUTED_AUDIT_PENDING":
        raise RuntimeError("full computation required")
    manifest = json.loads((args.input / "manifest.json").read_text())
    datasets = {}
    for seed in manifest["seeds"]:
        paths = sorted(args.input.glob(f"author_{seed}_application_*.json.gz"))
        if len(paths) != manifest["roles_per_seed"]["application"]:
            raise RuntimeError("missing application authors")
        packets = []
        for path in paths:
            with gzip.open(path, "rt") as stream:
                packets.append(json.load(stream))
        datasets[seed] = packets
    summaries, seed_rows, comparisons = [], [], []
    all_designs = manifest["designs"]
    for design_index, design in enumerate(all_designs):
        key = {name: design[name] for name in ("world", "variant", "level", "budget")}
        combined = []
        for seed, packets in datasets.items():
            rows = [p["rows"][design_index] for p in packets]
            combined.extend(rows)
            for target_index, target in enumerate(("a", "b")):
                seed_rows.append(
                    {
                        **key,
                        "seed": seed,
                        "target": target,
                        "mse": float(
                            np.mean([r["squared_error"][target_index] for r in rows])
                        ),
                        "coverage95": float(
                            np.mean([r["coverage95"][target_index] for r in rows])
                        ),
                    }
                )
        for j, target in enumerate(("a", "b")):
            coverage = np.mean([r["coverage95"][j] for r in combined])
            n = len(combined)
            denominator = 1 + 1.96**2 / n
            center = (coverage + 1.96**2 / (2 * n)) / denominator
            radius = (
                1.96
                * np.sqrt(coverage * (1 - coverage) / n + 1.96**2 / (4 * n * n))
                / denominator
            )
            summaries.append(
                {
                    **key,
                    "target": target,
                    "n_application": n,
                    "mse": float(np.mean([r["squared_error"][j] for r in combined])),
                    "bias": float(np.mean([r["error"][j] for r in combined])),
                    "coverage95": float(coverage),
                    "coverage_wilson_low": float(center - radius),
                    "coverage_wilson_high": float(center + radius),
                    "mean_interval_width": float(
                        np.mean(
                            [3.92 * np.sqrt(r["covariance"][j][j]) for r in combined]
                        )
                    ),
                    "optimizer_non_success": sum(not r["success"] for r in combined),
                }
            )
        if design["level"] in (None, 1):
            continue
        reference_index = next(
            i
            for i, other in enumerate(all_designs)
            if other["world"] == design["world"]
            and other["variant"] == design["variant"]
            and other["budget"] == design["budget"]
            and other["level"] == 1
        )
        for j, target in enumerate(("a", "b")):
            rng = np.random.default_rng(20260915)
            bootstraps, means = [], []
            for packets in datasets.values():
                delta = np.array(
                    [
                        p["rows"][design_index]["squared_error"][j]
                        - p["rows"][reference_index]["squared_error"][j]
                        for p in packets
                    ]
                )
                means.append(delta.mean())
                bootstraps.append(
                    delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(
                        axis=1
                    )
                )
            low, high = np.quantile(np.mean(bootstraps, axis=0), [0.025, 0.975])
            comparisons.append(
                {
                    **key,
                    "target": target,
                    "reference_level": 1,
                    "mse_delta": float(np.mean(means)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "status": "SUPPORTED_LOWER_MSE"
                    if high < 0
                    else "SUPPORTED_HIGHER_MSE"
                    if low > 0
                    else "INCONCLUSIVE",
                }
            )
    args.output.mkdir(parents=True, exist_ok=False)
    save_csv(args.output / "application_metrics.csv", summaries)
    save_csv(args.output / "metrics_by_seed.csv", seed_rows)
    save_csv(args.output / "paired_level1_comparisons.csv", comparisons)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "status": "AGGREGATED_REPLAY_PENDING",
                "metric_rows": len(summaries),
                "seed_rows": len(seed_rows),
                "paired_intervals": len(comparisons),
                "bootstrap": "2000 within-seed author resamples, seed20260915; level1 reference fixed for all designs",
                "scope": "all estimates retained including optimizer non-success; no application-based selection",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

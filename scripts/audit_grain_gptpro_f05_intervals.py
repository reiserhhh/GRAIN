#!/usr/bin/env python3
"""Reconstruct F05 paired intervals independently from stored author losses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256


def main():
    """Check exact protocol resampling without interpreting shared episodes as IID."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--missing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior evidence")
    args.output.mkdir(parents=True)
    largest = 0.0
    records = []
    for kind, directory in (
        ("application", args.application),
        ("missing", args.missing),
    ):
        for item in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            assert sha256(directory / item["path"]) == item["sha256"]
        data = pd.read_csv(
            directory
            / (
                "state_parameter_metrics.csv"
                if kind == "application"
                else "missing_time_metrics.csv"
            )
        )
        intervals = pd.read_csv(directory / "paired_comparisons.csv")
        for row in intervals.itertuples(index=False):
            subset = data[data.world == row.world]
            if kind == "application":
                subset = subset[subset.n == 48]
                column, left, right = "method", "personal_MAP", "source_only_dynamics"
            else:
                subset = subset[
                    (subset.budget == row.budget) & (subset.estimator == row.estimator)
                ]
                column, left, right = "clock", "correct_gap3", "wrong_gap1"
            rng = np.random.default_rng(20260915)
            means = []
            sampled = []
            for seed, group in subset.groupby("seed", sort=True):
                pairs = group.pivot(
                    index="author_id", columns=column, values=row.metric
                ).sort_index()
                differences = (pairs[left] - pairs[right]).to_numpy()
                means.append(float(np.mean(differences)))
                indices = rng.integers(0, len(differences), (2000, len(differences)))
                sampled.append(np.sum(differences[indices], axis=1) / len(differences))
                records.append(
                    {
                        "packet": kind,
                        "world": row.world,
                        "metric": row.metric,
                        "seed": int(seed),
                        "budget": 48 if kind == "application" else int(row.budget),
                        "estimator": left if kind == "application" else row.estimator,
                        "delta": means[-1],
                        "authors": len(differences),
                    }
                )
            low, high = np.quantile(
                np.sum(sampled, axis=0) / len(sampled), [0.025, 0.975]
            )
            expected = np.array([np.mean(means), low, high])
            actual = np.array([row.delta, row.ci_low, row.ci_high])
            np.testing.assert_allclose(expected, actual, atol=1e-10, rtol=1e-8)
            largest = max(largest, float(np.max(np.abs(expected - actual))))
    pd.DataFrame(records).to_csv(args.output / "seed_differences.csv", index=False)
    report = {
        "status": "EXACT_PAIRED_REPLAY_PASS",
        "intervals": len(records) // 3,
        "seed_differences": len(records),
        "max_absolute_difference": largest,
        "interpretation": "author resampling is conditional/descriptive for shared-shock streams; only three independent episodes. No new population CI claim.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == "__main__":
    main()

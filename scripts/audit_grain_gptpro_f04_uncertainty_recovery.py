#!/usr/bin/env python3
"""Replay F04 seed-stratified paired intervals without production bootstrap code."""

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
    """Rebuild all stored contrasts and tail means from author-level outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve audits")
    args.output.mkdir(parents=True)
    for directory in (args.application, args.tail):
        for item in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            if sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError("changed input")
    data = pd.read_csv(args.application / "response_metrics.csv", dtype={"world": str})
    data = data[(data.role == "application") & (data.K == 4) & (data.n == 24)]
    contrasts = pd.read_csv(
        args.application / "paired_comparisons.csv", dtype={"world": str}
    )
    largest = 0.0
    for row in contrasts.itertuples(index=False):
        subset = data[data.world == row.world]
        rng = np.random.default_rng(20260915)
        resamples, means = [], []
        for _, group in subset.groupby("seed", sort=True):
            pairs = group.pivot(
                index="author_id", columns="method", values=row.metric
            ).sort_index()
            delta = (pairs[row.method] - pairs.personal_OLS).to_numpy()
            index = rng.integers(0, len(delta), (2000, len(delta)))
            # Explicit replicate sums, not the production helper.
            resamples.append(np.sum(delta[index], axis=1) / len(delta))
            means.append(np.sum(delta) / len(delta))
        low, high = np.quantile(
            np.sum(resamples, axis=0) / len(resamples), [0.025, 0.975]
        )
        expected = [np.mean(means), low, high]
        actual = [row.delta, row.ci_low, row.ci_high]
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-8)
        largest = max(largest, float(np.max(np.abs(np.array(actual) - expected))))
    tails = pd.read_csv(args.tail / "variance_tail_by_author.csv", dtype={"world": str})
    aggregates = pd.read_csv(
        args.tail / "variance_tail_summary.csv", dtype={"world": str}
    )
    keys = ["world", "n", "mode"]
    checks = 0
    for row in aggregates.to_dict("records"):
        group = tails[
            (tails.world == row["world"])
            & (tails.n == row["n"])
            & (tails["mode"] == row["mode"])
        ]
        assert len(group) == row["authors"]
        for metric, value in row.items():
            if metric in keys + ["authors"]:
                continue
            expected = group[metric].mean()
            np.testing.assert_allclose(
                value, expected, atol=1e-10, rtol=1e-8, equal_nan=True
            )
            if np.isfinite(expected):
                largest = max(largest, abs(value - expected))
            checks += 1
    summary = {
        "status": "REPLAY_PASS",
        "paired_intervals": len(contrasts),
        "tail_aggregate_checks": checks,
        "max_absolute_difference": largest,
        "scope": "same-agent independent arithmetic, not independent scientific acceptance",
        "S_dependence_note": "Original author-bootstrap replayed exactly. S authors share intercepts in pairs; pair-cluster sensitivity remains desirable before population CI interpretation.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

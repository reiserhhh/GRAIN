#!/usr/bin/env python3
"""Independently replay F04 stored predictions, losses and aggregate arithmetic."""

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
from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, grouped
from grain_research.gptpro_foundation_v3.f04_recovery import generate_response_world


def main():
    """Use predictions as fixed inputs; recompute all target and future losses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior audits")
    args.output.mkdir(parents=True)
    hashes = 0
    for directory in (args.source, args.application, args.tail):
        for item in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            if sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError(f"artifact mismatch: {directory / item['path']}")
            hashes += 1
    data = pd.read_csv(args.application / "response_metrics.csv", dtype={"world": str})
    summary = json.loads((args.source / "summary.json").read_text())
    largest = 0.0
    checked = 0

    def check(actual, expected):
        nonlocal largest, checked
        if not np.allclose(actual, expected, atol=1e-10, rtol=1e-8, equal_nan=True):
            raise AssertionError((actual, expected))
        if np.isfinite(actual) and np.isfinite(expected):
            largest = max(largest, abs(actual - expected))
        checked += 1

    for outcome in summary["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        root = args.source / f"{world}_{seed}"
        truth = {
            row["author_id"]: row
            for row in json.loads((root / "truth.json").read_text())
        }
        for k in (1, 2, 4):
            if k == 4:
                observations = [
                    json.loads(line)
                    for role in ("validation", "application")
                    for line in (root / f"{role}.jsonl").read_text().splitlines()
                ]
            else:
                observations, _ = generate_response_world(
                    world,
                    seed,
                    counts={**summary["counts"], "source": 0},
                    events=summary["events"],
                    condition_count=k,
                )
            people = grouped(observations)
            subset = data[(data.world == world) & (data.seed == seed) & (data.K == k)]
            for row in subset.itertuples(index=False):
                means = np.asarray(json.loads(row.means), dtype=float)
                target = truth[row.author_id]
                true_means = np.asarray(target["means"])
                full = np.isfinite(means).all()
                if bool(row.complete_profile) != full:
                    raise AssertionError("unsupported profile was accepted")
                check(
                    row.profile_MSE,
                    np.mean((means - true_means) ** 2) if full else np.nan,
                )
                check(
                    row.delta_squared_error,
                    ((means[3] - means[0]) - (true_means[3] - true_means[0])) ** 2
                    if full
                    else np.nan,
                )
                check(
                    row.level_squared_error,
                    (means.mean() - true_means.mean()) ** 2 if full else np.nan,
                )
                linear = world in {"L", "V", "S", "0", "H"}
                check(
                    row.a_squared_error,
                    (means.mean() - target["a"]) ** 2 if full and linear else np.nan,
                )
                check(
                    row.b_squared_error,
                    ((means[3] - means[0]) / 2 - target["b"]) ** 2
                    if full and linear
                    else np.nan,
                )
                future = people[row.author_id][row.n :]
                check(row.future_n, len(future))
                errors = [
                    (event["y"] - means[list(CONDITIONS).index(event["z"])]) ** 2
                    for event in future
                ]
                check(row.future_MSE, np.mean(errors) if errors else np.nan)
    metrics = [
        "future_MSE",
        "profile_MSE",
        "delta_squared_error",
        "level_squared_error",
        "a_squared_error",
        "b_squared_error",
    ]
    grouped_means = (
        data[data.role == "application"]
        .groupby(["world", "K", "n", "method"])[metrics]
        .mean()
    )
    stored = pd.read_csv(
        args.application / "aggregate_metrics.csv", dtype={"world": str}
    ).set_index(["world", "K", "n", "method"])
    for key, values in grouped_means.iterrows():
        for metric in metrics:
            check(stored.loc[key, metric], values[metric])
    tails = pd.read_csv(args.tail / "variance_tail_by_author.csv", dtype={"world": str})
    if set(tails.loc[tails.world == "H", "interval"]) != {"bootstrap"}:
        raise AssertionError("heavy-tail chi-square interval")
    for row in tails.itertuples(index=False):
        check(row.variance, row.rss / row.df)
        check(row.variance_squared_error, (row.variance - row.true_variance) ** 2)
        check(row.operational_threshold, 2 * np.sqrt(row.variance))
    report = {
        "status": "NUMERICAL_REPLAY_PASS",
        "hashes_verified": hashes,
        "prediction_rows": len(data),
        "tail_rows": len(tails),
        "scalar_checks": checked,
        "max_absolute_difference": largest,
        "scope": "same-agent alternative arithmetic audit; not independent scientific review or full F04 card acceptance",
        "remaining": [
            "paired bootstrap replay",
            "analytic information and MC SE closure",
            "source/DAG card packaging",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

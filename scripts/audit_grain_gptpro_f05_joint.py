#!/usr/bin/env python3
"""Replay joint-channel arithmetic and equality to previous independent filters."""

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
    """Check matched marginal outputs before interpreting off-diagonal changes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--application", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior audit")
    args.output.mkdir(parents=True)
    hashes = 0
    for directory in (args.joint, args.application):
        for item in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            assert sha256(directory / item["path"]) == item["sha256"]
            hashes += 1
    data = pd.read_csv(args.joint / "joint_state_metrics.csv")
    previous = pd.read_csv(args.application / "state_parameter_metrics.csv")
    independent = data[data.arm == "independent"].copy()
    independent["n"] = independent.t + 1
    independent["method"] = independent.method.replace({"oracle": "oracle_Kalman"})
    keys = ["world", "seed", "author_id", "n", "method"]
    merged = independent.merge(
        previous, on=keys, validate="one_to_one", suffixes=("_joint", "_old")
    )
    assert len(merged) == len(independent)
    largest = 0.0
    for metric in ("state_mean", "state_variance"):
        np.testing.assert_allclose(
            merged[metric + "_joint"], merged[metric + "_old"], atol=1e-10, rtol=1e-8
        )
        largest = max(
            largest,
            float(np.max(np.abs(merged[metric + "_joint"] - merged[metric + "_old"]))),
        )
    np.testing.assert_allclose(
        data.state_MSE, (data.state_mean - data.state_truth) ** 2, atol=1e-10, rtol=1e-8
    )
    expected = (
        np.abs(data.state_mean - data.state_truth)
        <= 1.959963984540054 * np.sqrt(data.state_variance)
    ).astype(float)
    np.testing.assert_array_equal(data.state_coverage, expected)
    future = pd.read_csv(args.application / "forecast_metrics.csv")
    count = 0
    for k in (1, 4):
        original = future[future.k == k]
        subset = data[data[f"forecast{k}"].notna()].copy()
        subset["n"] = subset.t + 1
        subset["method"] = subset.method.replace({"oracle": "oracle_Kalman"})
        joined = subset.merge(
            original, on=keys, validate="many_to_one", suffixes=("_joint", "_old")
        )
        assert len(joined) == len(subset)
        for metric, target in (
            (f"forecast{k}_MSE", "future_y"),
            (f"conditional{k}_MSE", "conditional_truth"),
        ):
            computed = (joined[f"forecast{k}"] - joined[target]) ** 2
            np.testing.assert_allclose(joined[metric], computed, atol=1e-10, rtol=1e-8)
        count += len(joined)
    report = {
        "status": "MARGINAL_EQUIVALENCE_AND_METRICS_PASS",
        "rows": len(data),
        "marginal_rows": len(merged),
        "marginal_max_error": largest,
        "forecast_rows": count,
        "hashes": hashes,
        "scope": "full arithmetic and previous independent-filter equality; joint multivariate law separately checked by exact small-panel Gaussian tests",
        "remaining": "original task review; no population CI from only three episodes",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == "__main__":
    main()

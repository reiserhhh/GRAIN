#!/usr/bin/env python3
"""Replay F05 application arithmetic and repair boolean aggregate coercion."""

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


def numeric_coverage(values):
    """Convert nullable boolean indicators explicitly, avoiding object reductions."""
    return values.map(
        lambda v: (
            np.nan if pd.isna(v) else float(v is True or str(v) in {"True", "1", "1.0"})
        )
    ).astype(float)


def main():
    """Recompute from frozen individual outputs; no predictor rerun or new fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve audits")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.application / "output_manifest.json").read_text())
    for item in manifest["files"]:
        assert sha256(args.application / item["path"]) == item["sha256"]
    state = pd.read_csv(args.application / "state_parameter_metrics.csv")
    forecast = pd.read_csv(args.application / "forecast_metrics.csv")
    state["state_covered"] = numeric_coverage(state.state_covered)
    state_error = (state.state_mean - state.state_truth) ** 2
    np.testing.assert_allclose(
        state.state_squared_error, state_error, atol=1e-10, rtol=1e-8
    )
    mask = state.state_variance.notna()
    coverage = (
        np.abs(state.state_mean - state.state_truth)
        <= 1.959963984540054 * np.sqrt(state.state_variance)
    ).astype(float)
    np.testing.assert_array_equal(state.loc[mask, "state_covered"], coverage[mask])
    for target, metric in (
        ("future_y", "forecast_squared_error"),
        ("conditional_truth", "conditional_squared_error"),
    ):
        expected = (forecast.prediction - forecast[target]) ** 2
        np.testing.assert_allclose(forecast[metric], expected, atol=1e-10, rtol=1e-8)
    keys = ["world", "n", "phase", "method"]
    metrics = [
        "a_squared_error",
        "b_squared_error",
        "rho_squared_error",
        "state_squared_error",
        "state_covered",
    ]
    aggregate = state.groupby(keys)[metrics].mean().reset_index()
    # Python scalar summation separately checks the repaired coverage numerator.
    for key, group in state.groupby(keys):
        valid = [float(v) for v in group.state_covered if np.isfinite(v)]
        if valid:
            selected = aggregate
            for column, value in zip(keys, key):
                selected = selected[selected[column] == value]
            assert abs(selected.state_covered.iloc[0] - sum(valid) / len(valid)) < 1e-12
    aggregate.to_csv(args.output / "state_summary.csv", index=False)
    old = pd.read_csv(args.application / "state_summary.csv")
    comparison = aggregate.merge(old, on=keys, suffixes=("_repaired", "_old"))
    comparison["coverage_difference"] = (
        comparison.state_covered_repaired - comparison.state_covered_old
    )
    comparison.to_csv(args.output / "aggregate_correction.csv", index=False)
    for metric in metrics[:-1]:
        np.testing.assert_allclose(
            comparison[metric + "_repaired"],
            comparison[metric + "_old"],
            equal_nan=True,
            atol=1e-10,
            rtol=1e-8,
        )
    availability = state.groupby(keys).agg(
        authors=("author_id", "size"), successful_fits=("fit_success", "sum")
    )
    availability["fit_rate"] = availability.successful_fits / availability.authors
    availability.to_csv(args.output / "fit_availability.csv")
    state[state.fit_success].groupby(keys)[metrics].mean().reset_index().to_csv(
        args.output / "successful_fit_summary.csv", index=False
    )
    changed = int((comparison.coverage_difference.abs() > 1e-10).sum())
    report = {
        "status": "ARITHMETIC_PASS_COVERAGE_AGGREGATE_CORRECTED",
        "state_rows": len(state),
        "forecast_rows": len(forecast),
        "source_hashes_checked": len(manifest["files"]),
        "changed_coverage_cells": changed,
        "cause": "mixed numpy.bool_ and None created object-type group reductions; numeric 0/1 coercion restores arithmetic means",
        "unchanged": "author predictions, variance, fit outcomes and all squared errors; no refit",
        "remaining": "independent prediction/state-law replay, paired CI replay and shared-episode dependence audit",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "input_manifest.json").write_text(
        json.dumps(
            {
                "application_manifest_sha256": sha256(
                    args.application / "output_manifest.json"
                )
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()

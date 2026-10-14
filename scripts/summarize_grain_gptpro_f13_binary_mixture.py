#!/usr/bin/env python3
"""Author-cluster descriptive inference for frozen F08 uncertainty outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Do not count sequential events as independent calibration trials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(args.input / "predictions.csv.gz")
    keys = ["world", "seed", "panel", "channel"]
    assert not data.duplicated(keys + ["author_id", "t"]).any()
    values = [
        "state_variance",
        "source_variance",
        "event_variance",
        "probability_squared_error",
        "brier",
        "working_probability_covered",
    ]
    authors = data.groupby(keys + ["author_id"])[values].mean().reset_index()
    authors.to_csv(args.output / "author_metrics.csv", index=False)
    rows = []
    rng = np.random.default_rng(20260915)
    for key, group in authors.groupby(keys):
        assert len(group) == 200
        array = group[values].to_numpy()
        indices = rng.integers(0, len(group), (2000, len(group)))
        means = array[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975], axis=0)
        for j, name in enumerate(values):
            rows.append(
                dict(zip(keys, key, strict=True))
                | {
                    "metric": name,
                    "mean": float(array[:, j].mean()),
                    "author_bootstrap_low": float(low[j]),
                    "author_bootstrap_high": float(high[j]),
                    "authors": 200,
                }
            )
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    identity = abs(data.outcome_variance - data["mean"] * (1 - data["mean"])).max()
    assert identity < 1e-12 and len(data) == 259200
    summary = {
        "status": "AUTHOR_CLUSTER_SUMMARY",
        "rows": len(data),
        "author_panel_channel_rows": len(authors),
        "max_identity_error": float(identity),
        "mc_mean_rmse": float(np.sqrt(np.mean((data.mc_mean - data["mean"]) ** 2))),
        "scope": "2000 author bootstrap samples within seed; coverage is descriptive for moment-based working probability intervals, not event-independent Wilson or a posterior for the unknown predictive mean",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

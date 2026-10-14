#!/usr/bin/env python3
"""Summarize completed history-imputation arms without pooling events as people."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Require all planned groups and report paired author-level differences."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.input / "summary.json").read_text())
    if status["status"] != "COMPLETE" or status["completed"] != 2400:
        raise ValueError("full application output is not complete")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for i in range(2400):
        payload = json.loads((args.input / f"group_{i:04d}.json").read_text())
        records.extend(
            r | {"author_id": payload["author_id"], "panel": payload["panel"]}
            for r in payload["rows"]
        )
    data = pd.DataFrame(records)
    keys = ["world", "seed", "panel", "channel", "head"]
    assert len(data) == 518400 and not data.duplicated(keys + ["author_id", "t"]).any()
    data["online_mse"] = (data.online - data.true_next_p) ** 2
    data["imputation_mse"] = (data["mean"] - data.true_next_p) ** 2
    data["delta_mse"] = data.imputation_mse - data.online_mse
    data["absolute_mean_shift"] = abs(data["mean"] - data.online)
    data["clipped_fraction"] = data.clip_count / 200
    columns = [
        "online_mse",
        "imputation_mse",
        "delta_mse",
        "absolute_mean_shift",
        "history_refit_variance",
        "event_variance",
        "clipped_fraction",
    ]
    authors = data.groupby(keys + ["author_id"])[columns].mean().reset_index()
    authors.to_csv(args.output / "author_metrics.csv", index=False)
    rng = np.random.default_rng(20260915)
    metrics = []
    for key, group in authors.groupby(keys):
        assert len(group) == 200
        values = group[columns].to_numpy()
        bootstrap = values[rng.integers(0, 200, (2000, 200))].mean(axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for j, column in enumerate(columns):
            metrics.append(
                dict(zip(keys, key, strict=True))
                | {
                    "metric": column,
                    "mean": float(values[:, j].mean()),
                    "low": float(low[j]),
                    "high": float(high[j]),
                }
            )
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    summary = {
        "status": "FULL_PAIRED_AUTHOR_SUMMARY",
        "rows": len(data),
        "author_head_channel_panel_rows": len(authors),
        "scope": "fixed-source history imputation vs frozen online estimator; not total parameter uncertainty",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

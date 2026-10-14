#!/usr/bin/env python3
"""Replay the chart with scalar polynomial arithmetic, not production functions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def scalar_value(e, h0, a, u, b, q):
    """Evaluate the polynomial by explicit scalar summation."""
    h = [h0[i] + sum(a[i, j] * e[j] for j in range(4)) for i in range(4)]
    return sum(u[q, i] * h[i] for i in range(4)) + 0.5 * sum(
        b[q, i, j] * h[i] * h[j] for i in range(4) for j in range(4)
    )


def main():
    """Check all predictions, source covariance and generalized eigen certificates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    worlds = {}
    eigen_error = 0.0
    for path in sorted(args.input.glob("world_*.npz")):
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        seed, world = path.stem.removeprefix("world_").split("_", 1)
        worlds[int(seed), world] = data
        centered = data["edits"][:300] - data["edits"][:300].mean(axis=0)
        np.testing.assert_allclose(
            centered.T @ centered / 299, data["source_covariance"], atol=1e-12
        )
        v, m, j = data["action_basis"], data["metric"], data["j"]
        g = (np.outer(j[0], j[0]) + np.outer(j[1], j[1])) / 2
        values = np.diag(v.T @ g @ v)
        eigen_error = max(
            eigen_error, float(np.max(np.abs(g @ v - m @ v @ np.diag(values))))
        )
        np.testing.assert_allclose(v.T @ m @ v, np.eye(2), atol=1e-12)
    rows = list(csv.DictReader((args.input / "predictions.csv").open()))
    error = 0.0
    for row in rows:
        d = worlds[int(row["seed"]), row["world"]]
        e = d["edits"][int(row["index"])]
        method = row["method"]
        if method == "zero":
            reconstructed = np.zeros(4)
        elif method == "full_chart":
            reconstructed = e
        else:
            v = d["pca"] if method == "pca_rank2" else d["action_basis"]
            m = np.eye(4) if method == "pca_rank2" else d["metric"]
            # Whiten the metric and solve the projection as least squares.
            root = np.linalg.cholesky(m).T
            reconstructed = v @ np.linalg.lstsq(root @ v, root @ e, rcond=None)[0]
        q = int(row["query"])
        true = scalar_value(e, d["h0"], d["a"], d["u"], d["b"], q)
        predicted = scalar_value(reconstructed, d["h0"], d["a"], d["u"], d["b"], q)
        error = max(
            error,
            abs(true - float(row["truth"])),
            abs(predicted - float(row["prediction"])),
        )
    metric_error = 0.0
    for row in csv.DictReader((args.input / "metrics.csv").open()):
        subset = [
            r for r in rows if all(r[k] == row[k] for k in ("world", "role", "method"))
        ]
        for prefix in ("", "linear_"):
            mse = sum(
                (float(r[prefix + "truth"]) - float(r[prefix + "prediction"])) ** 2
                for r in subset
            ) / len(subset)
            metric_error = max(
                metric_error,
                abs(mse - float(row["linear_mse" if prefix else "full_mse"])),
            )
    if max(error, metric_error, eigen_error) > 1e-9:
        raise ValueError("independent arithmetic replay failed")
    result = {
        "status": "PASS_NUMERICAL_REPLAY",
        "rows": len(rows),
        "max_score_difference": error,
        "max_metric_difference": metric_error,
        "max_eigen_difference": eigen_error,
        "independent_agent": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

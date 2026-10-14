#!/usr/bin/env python3
"""Alternate source moment reconstruction, REML refits and metric replay."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def source_basis():
    """Build joint covariance coefficients by scalar cell identities."""
    cells = [(k, r, t) for k in range(3) for r in range(3) for t in range(2)]
    pairs = [(i, j) for i in range(3) for j in range(i + 1)]
    basis = np.zeros((21 * 21, 9))
    for i in range(21):
        k = cells[i][0] if i < 18 else i - 18
        for j in range(21):
            kk = cells[j][0] if j < 18 else j - 18
            basis[21 * i + j, pairs.index((max(k, kk), min(k, kk)))] = 1
            if i < 18 and j < 18:
                basis[21 * i + j, 6] = cells[i][1] == cells[j][1]
                basis[21 * i + j, 7] = i == j
                basis[21 * i + j, 8] = 1
    return basis


def objective(theta, y, conditional, event):
    """Scalar covariance plus slogdet/inverse REML, no production function call."""
    l = np.zeros((3, 3))
    l[np.tril_indices(3)] = theta[:6]
    l[np.diag_indices(3)] = np.exp(l.diagonal())
    s = l @ l.T
    method, error = np.exp(theta[6:8])
    shared = np.exp(theta[8]) if event else 0.0
    cells = [(k, r, t) for k in range(3) for r in range(3) for t in range(2)]
    load = [1 + 0.3 * (r - 1) * (2 * t - 1) if conditional else 1 for k, r, t in cells]
    v = np.array(
        [
            [
                load[i] * load[j] * s[k, kk]
                + method * (r == rr)
                + shared * (t == tt)
                + error * (i == j)
                for j, (kk, rr, tt) in enumerate(cells)
            ]
            for i, (k, r, t) in enumerate(cells)
        ]
    )
    centered = y - y.mean(axis=0)
    sign, logdet = np.linalg.slogdet(v)
    if sign <= 0:
        return 1e100
    return 0.5 * (
        (len(y) - 1) * logdet + np.sum((centered @ np.linalg.inv(v)) * centered)
    )


def main():
    """Replay source fits and all aggregate MTMM metrics without changing results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtmm", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    q, r = np.linalg.qr(source_basis(), mode="reduced")
    anchor_error = 0.0
    for path in args.anchor.glob("fit_*_source_anchored.json"):
        seed, world, _ = path.stem.removeprefix("fit_").split("_", 2)
        with np.load(args.anchor / f"observations_{seed}_source.npz") as z:
            observations = np.column_stack([z["text"], z[f"anchor_{world}"]])
        centered = observations - observations.mean(axis=0)
        cov = centered.T @ centered / (len(observations) - 1)
        cov[18:, 18:] -= 0.25 * np.eye(3)
        coefficients = np.linalg.solve(r, q.T @ cov.ravel())
        stored = json.loads(path.read_text())
        anchor_error = max(
            anchor_error,
            float(np.max(np.abs(coefficients - stored["raw_coefficients"]))),
        )
    refits = []
    for path in sorted(args.mtmm.glob("fit_*_reml.json")):
        seed, world = path.stem.removeprefix("fit_").removesuffix("_reml").split("_", 1)
        saved = json.loads(path.read_text())
        with np.load(args.mtmm / f"data_{seed}_{world}_source.npz") as z:
            y = z["observed"]
        theta = np.array(saved["theta"])
        fun = lambda x, y=y, world=world: objective(
            x, y, world == "conditional_loading", world != "base"
        )
        before = fun(theta)
        bounds = [(-12, 8) if i in (0, 2, 5) else (-20, 20) for i in range(6)] + [
            (-18, 8)
        ] * (len(theta) - 6)
        result = minimize(
            fun,
            theta,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 100, "xtol": 1e-9, "ftol": 1e-12},
        )
        refits.append(
            {
                "seed": seed,
                "world": world,
                "stored_objective": saved["objective"],
                "replayed_objective": before,
                "powell_objective": float(result.fun),
                "improvement": before - float(result.fun),
                "success": bool(result.success),
                "calls": result.nfev,
            }
        )
    metric_error = 0.0
    with (args.mtmm / "metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    for row in metrics:
        stem = f"{row['seed']}_{row['world']}"
        with np.load(args.mtmm / f"truth_{stem}_{row['role']}.npz") as z:
            truth = z["trait"]
        with np.load(
            args.mtmm / f"pred_{stem}_{row['method']}_{row['role']}_{row['panel']}.npz"
        ) as z:
            p, v = z["prediction"], z["variance"]
        values = {
            "mse": np.mean((p - truth) ** 2),
            "coverage": np.mean(np.abs(p - truth) <= 1.96 * np.sqrt(np.diag(v))),
            "mean_variance": np.mean(np.diag(v)),
        }
        metric_error = max(
            metric_error, max(abs(float(row[k]) - value) for k, value in values.items())
        )
    with (args.output / "optimizer_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(refits[0]))
        writer.writeheader()
        writer.writerows(refits)
    if anchor_error > 1e-10 or metric_error > 1e-10:
        raise ValueError("source/metric replay failed")
    summary = {
        "status": "SOURCE_QR_AND_AGGREGATE_REPLAY_COMPLETE",
        "anchor_fits": 6,
        "max_anchor_coefficient_difference": anchor_error,
        "metric_rows": len(metrics),
        "max_metric_error": metric_error,
        "reml_checks": len(refits),
        "powell_success": sum(x["success"] for x in refits),
        "maximum_objective_improvement": max(x["improvement"] for x in refits),
        "scope": "local alternate optimizer check, not proof of global optimum; original fits unchanged",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

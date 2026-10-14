#!/usr/bin/env python3
"""Run frozen four-edit quadratic worlds; do not label them native LLM paths."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.gptpro_foundation_v3.f10_geometry import (
    action_subspace,
    edit_jacobian,
    project_edits,
    quadratic_response,
)


def main():
    """Compare source-only rank-two PCA and action spaces without tuning dimensions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "seeds": [20260915, 20260916, 20260917],
        "source": 300,
        "validation": 100,
        "application": 200,
        "edit_coordinates": ["content_pole", "subject", "modality", "audience"],
        "covariance": [0.04, 0.09, 4.0, 9.0],
        "dimension": 2,
        "worlds": ["linear", "null_curvature"],
        "scope": "known continuous synthetic chart, not native text attention",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, checks, summaries = [], [], []
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        edits = rng.normal(size=(600, 4)) * np.sqrt(config["covariance"])
        c = np.cov(edits[:300], rowvar=False)
        h0, a = np.zeros(4), np.eye(4)
        u = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        _, pca = np.linalg.eigh(c)
        pca = pca[:, -2:]
        for world in config["worlds"]:
            b = np.zeros((2, 4, 4))
            if world == "null_curvature":
                b[0, 2, 2], b[1, 3, 3] = 0.1, 0.05
            j = edit_jacobian(np.zeros(4), h0, a, u, b)
            geometry = action_subspace(j, c)
            v, m = geometry["basis"], geometry["denominator"]
            np.savez(
                args.output / f"world_{seed}_{world}.npz",
                edits=edits,
                source_covariance=c,
                h0=h0,
                a=a,
                u=u,
                b=b,
                j=j,
                action_basis=v,
                metric=m,
                pca=pca,
            )
            for index, e in enumerate(edits[:300]):
                exact = edit_jacobian(e, h0, a, u, b)
                for step in (1e-2, 1e-3, 1e-4):
                    numerical = np.column_stack(
                        [
                            (
                                quadratic_response(e + d, h0, a, u, b)
                                - quadratic_response(e - d, h0, a, u, b)
                            )
                            / (2 * step)
                            for d in step * np.eye(4)
                        ]
                    )
                    error = float(np.max(np.abs(numerical - exact)))
                    checks.append(
                        {
                            "seed": seed,
                            "world": world,
                            "source_index": index,
                            "step": step,
                            "max_error": error,
                        }
                    )
                    if error > 1e-5:
                        raise ValueError("quadratic derivative mismatch")
            observed = edits[300:]
            candidates = {
                "zero": np.zeros_like(observed),
                "pca_rank2": project_edits(observed, pca),
                "action_rank2": project_edits(observed, v, m),
                "full_chart": observed,
            }
            full = quadratic_response(observed, h0, a, u, b)
            linear = observed @ j.T
            for method, reconstructed in candidates.items():
                predicted = quadratic_response(reconstructed, h0, a, u, b)
                linear_pred = reconstructed @ j.T
                for k in range(300):
                    role = "validation" if k < 100 else "application"
                    for q in range(2):
                        rows.append(
                            {
                                "seed": seed,
                                "world": world,
                                "role": role,
                                "index": k + 300,
                                "query": q,
                                "method": method,
                                "truth": float(full[k, q]),
                                "prediction": float(predicted[k, q]),
                                "linear_truth": float(linear[k, q]),
                                "linear_prediction": float(linear_pred[k, q]),
                            }
                        )
            eig_error = np.max(
                np.abs(geometry["metric"] @ v - m @ v @ np.diag(geometry["values"][:2]))
            )
            summaries.append(
                {
                    "seed": seed,
                    "world": world,
                    "rank": geometry["rank"],
                    "null_dimension": geometry["nullspace"].shape[1],
                    "generalized_eigen_error": float(eig_error),
                }
            )
    metrics = []
    for world in config["worlds"]:
        for role in ("validation", "application"):
            for method in ("zero", "pca_rank2", "action_rank2", "full_chart"):
                subset = [
                    r
                    for r in rows
                    if (r["world"], r["role"], r["method"]) == (world, role, method)
                ]
                metrics.append(
                    {
                        "world": world,
                        "role": role,
                        "method": method,
                        "n": len(subset),
                        "full_mse": float(
                            np.mean(
                                [(r["truth"] - r["prediction"]) ** 2 for r in subset]
                            )
                        ),
                        "linear_mse": float(
                            np.mean(
                                [
                                    (r["linear_truth"] - r["linear_prediction"]) ** 2
                                    for r in subset
                                ]
                            )
                        ),
                    }
                )
    for name, data in (
        ("predictions.csv", rows),
        ("derivative_checks.csv", checks),
        ("metrics.csv", metrics),
        ("geometry.csv", summaries),
    ):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "status": "SYNTHETIC_ACTION_AND_CURVATURE_COMPUTED",
        "derivative_checks": len(checks),
        "predictions": len(rows),
        "max_derivative_error": max(r["max_error"] for r in checks),
        "max_eigen_error": max(r["generalized_eigen_error"] for r in summaries),
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_J_claim": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

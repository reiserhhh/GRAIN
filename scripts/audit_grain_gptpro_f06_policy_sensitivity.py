#!/usr/bin/env python3
"""Backward path integration independently checks F06 forward grid likelihoods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256


def backward_integrals(rows, a, b, rho, beta, q, r, limit, points):
    """Integrate the terminal 1/s/s^2 functions backward over the whole path."""
    grid = np.linspace(-limit, limit, points)
    dx = 2 * limit / (points - 1)
    quadrature = np.full(points, dx)
    quadrature[[0, -1]] /= 2
    prior_v = q / (1 - rho**2)
    prior = np.exp(-(grid**2) / (2 * prior_v)) * quadrature
    prior /= prior.sum()
    kernels = {}
    for h in (0, 1):
        difference = grid[:, None] - rho * grid[None, :] - beta * h
        kernels[h] = (
            np.exp(-(difference**2) / (2 * q))
            * quadrature[:, None]
            / np.sqrt(2 * np.pi * q)
        )
    terminal = np.column_stack([np.ones(points), grid, grid**2])
    message = np.tile(terminal[:, None, :], (1, 3, 1))
    scales = np.zeros(3)
    for t in range(len(rows) - 1, -1, -1):
        row = rows[t]
        emission = -0.5 * (
            (row["y"] - a - b * row["z"] - grid) ** 2 / r + np.log(2 * np.pi * r)
        )
        potential = emission[:, None] + log_expit(
            (2 * row["h"] - 1) * grid[:, None] * np.array([0, 1, 3])[None, :]
        )
        offset = potential.max(axis=0)
        message *= np.exp(potential - offset)[:, :, None]
        scale = message[:, :, 0].max(axis=0)
        message /= scale[None, :, None]
        scales += offset + np.log(scale)
        if t:
            message = (
                kernels[rows[t - 1]["h"]].T @ message.reshape(points, -1)
            ).reshape(points, 3, 3)
    integral = np.einsum("i,ijk->jk", prior, message)
    mean = integral[:, 1] / integral[:, 0]
    return scales + np.log(integral[:, 0]), mean


def dense_gaussian(rows, a, b, rho, beta, q, r):
    """Whole-vector normal likelihood for a realized, causally known h sequence."""
    n = len(rows)
    mean = np.zeros(n)
    for t in range(1, n):
        mean[t] = rho * mean[t - 1] + beta * rows[t - 1]["h"]
    prior_v = q / (1 - rho**2)
    covariance = (
        prior_v * rho ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        + np.eye(n) * r
    )
    residual = np.array([x["y"] - a - b * x["z"] for x in rows]) - mean
    _, logdet = np.linalg.slogdet(covariance)
    solve = np.linalg.solve(covariance, residual)
    loglik = -0.5 * (n * np.log(2 * np.pi) + logdet + residual @ solve)
    cross = prior_v * rho ** (n - 1 - np.arange(n))
    return loglik, mean[-1] + cross @ solve


def main():
    """Audit all fixed-grid cells plus paired causal identities without refitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier audit")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.input / "output_manifest.json").read_text())["files"]
    for item in manifest:
        assert sha256(args.input / item["path"]) == item["sha256"]
    truths = json.loads((args.input / "truth.json").read_text())
    full = pd.read_csv(args.input / "all_profiles.csv")
    checked = 0
    maximum = 0.0
    analytic_grid_error = 0.0
    checks = []
    for case_name, data in full.groupby("case", sort=True):
        case = args.input / case_name
        rows = json.loads((case / "observed.json").read_text())
        source = json.loads((case / "source_model.json").read_text())
        fit = json.loads((case / "prefix_fit.json").read_text())
        a, b = fit["psi"][:2]
        q, r = np.exp(2 * np.asarray(source["log_sigma"]))
        truth = truths[case_name]
        delta = np.asarray(truth["delta_state"])
        h = np.array([x["h"] for x in rows])
        np.testing.assert_allclose(
            delta[1:],
            truth["rho"] * delta[:-1] + truth["beta"] * h[:-1],
            atol=1e-10,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.sum(delta**2), truth["sum_squared_perturbation"], atol=1e-10, rtol=1e-8
        )
        if truth["beta"] == 0:
            assert np.all(delta == 0)
        for (rho, beta), group in data.groupby(["rho", "beta"], sort=True):
            if not (group.status == "OK").all():
                raise AssertionError(
                    "unexpected failed cells require explicit failure replay"
                )
            if truth["policy"] == "hidden":
                group = group.sort_values("gamma")
                loglik, means = backward_integrals(
                    rows,
                    a,
                    b,
                    rho,
                    beta,
                    q,
                    r,
                    group.grid_limit.iloc[0],
                    401 if group.grid_limit.iloc[0] == 8 else 601,
                )
                baseline, _ = dense_gaussian(rows, a, b, rho, beta, q, r)
                analytic_grid_error = max(
                    analytic_grid_error,
                    abs(loglik[0] + len(rows) * np.log(2) - baseline),
                )
            else:
                loglik, means = dense_gaussian(rows, a, b, rho, beta, q, r)
                loglik, means = np.array([loglik]), np.array([means])
            expected = np.column_stack([loglik, means])
            actual = group[["loglik", "final_state"]].to_numpy()
            np.testing.assert_allclose(expected, actual, atol=1e-10, rtol=1e-8)
            maximum = max(maximum, float(np.max(np.abs(expected - actual))))
            checked += len(group)
        checks.append(
            {
                "case": case_name,
                "profile_rows": len(data),
                "prefix_fit_success": fit["success"],
                "sum_h": int(h.sum()),
                "sum_squared_perturbation": truth["sum_squared_perturbation"],
            }
        )
        print(json.dumps({"case": case_name, "checked": checked}), flush=True)
    assert checked == len(full)
    pd.DataFrame(checks).to_csv(args.output / "case_checks.csv", index=False)
    summary = {
        "status": "BACKWARD_PATH_AND_DENSE_GAUSSIAN_REPLAY_PASS",
        "profile_rows": checked,
        "cases": len(checks),
        "verified_hashes": len(manifest),
        "maximum_replay_difference": maximum,
        "gamma0_grid_vs_untruncated_analytic_difference": analytic_grid_error,
        "interpretation": "same-agent alternate direction integration; finite grid discrepancy is reported separately from same-grid arithmetic",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

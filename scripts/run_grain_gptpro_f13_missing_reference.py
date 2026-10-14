#!/usr/bin/env python3
"""Actual missingness/AR/population-shift references, not renamed biased means."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint

from grain_research.gptpro_foundation_v3.f13_history_contract import conditional_moments


def missing_probabilities(kind, z, y):
    """Operationalize frozen rates; MNAR depends on the unobserved outcome."""
    if kind == "complete":
        return np.zeros_like(y)
    if kind == "MCAR":
        return np.full_like(y, 0.2)
    if kind == "MAR":
        return np.broadcast_to(np.where(z < 0, 0.2, 0.6), y.shape)
    if kind == "MNAR":
        return 0.2 + 0.4 * expit(y)
    raise ValueError("unknown missingness mechanism")


def main():
    """Keep application truth out of score functions and distinguish oracle scope."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    n, events = (6, 8) if args.smoke else (200, 32)
    protocol = {
        "seeds": seeds,
        "authors": n,
        "events": events,
        "worlds": ["iid", "ar06", "prior_shift"],
        "masks": ["complete", "MCAR", "MAR", "MNAR"],
        "MCAR": 0.2,
        "MAR_negative_positive": [0.2, 0.6],
        "MNAR": "P(missing|y)=.2+.4*expit(y); explicit operational choice before run",
        "shift": "F12 old mean[0,.5],covdiag[1,.25];new mean[1,.2],covdiag[1.4,.7]",
        "arms": ["known_current_law", "old_iid_law"],
        "scope": "known Gaussian parameter reference, selection ignored in Gaussian update; MNAR arm is NOT an exact conditional posterior; no source bootstrap yet",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], events // 4)
    x = np.column_stack([np.ones(events), z])
    old_mean, old_prior = np.array([0.0, 0.5]), np.diag([1.0, 0.25])
    targets = {"level": np.array([1.0, 0.0]), "delta": np.array([0.0, 2.0])}
    rows = []
    for seed in seeds:
        parameters, errors, selection = np.random.SeedSequence([seed, 13, 4, 3]).spawn(
            3
        )
        raw_theta = np.random.default_rng(parameters).normal(size=(n, 2))
        raw_error = np.random.default_rng(errors).normal(size=(n, events))
        uniform = np.random.default_rng(selection).random((n, events))
        for world in protocol["worlds"]:
            mean = np.array([1.0, 0.2]) if world == "prior_shift" else old_mean
            prior = np.diag([1.4, 0.7]) if world == "prior_shift" else old_prior
            rho = 0.6 if world == "ar06" else 0.0
            noise = 0.25 * rho ** abs(np.arange(events)[:, None] - np.arange(events))
            theta = raw_theta @ np.linalg.cholesky(prior).T + mean
            y = theta @ x.T + raw_error @ np.linalg.cholesky(noise).T
            for kind in protocol["masks"]:
                probability = missing_probabilities(kind, z, y)
                observed = uniform >= probability
                np.savez_compressed(
                    args.output / f"{world}_{kind}_{seed}.npz",
                    y=y,
                    observed=observed,
                    missing_probability=probability,
                    theta=theta,
                    z=z,
                )
                for i in range(n):
                    keep = observed[i]
                    for arm in protocol["arms"]:
                        mu, v, r = (
                            (mean, prior, noise)
                            if arm == "known_current_law"
                            else (old_mean, old_prior, 0.25 * np.eye(events))
                        )
                        fitted = (
                            conditional_moments(
                                mu, v, x[keep], r[np.ix_(keep, keep)], y[i, keep]
                            )
                            if keep.any()
                            else {"mean": mu, "covariance": v}
                        )
                        for name, target in targets.items():
                            estimate = float(target @ fitted["mean"])
                            variance = float(target @ fitted["covariance"] @ target)
                            truth = float(target @ theta[i])
                            half = 1.96 * np.sqrt(max(0.0, variance))
                            rows.append(
                                {
                                    "world": world,
                                    "missingness": kind,
                                    "seed": seed,
                                    "author_index": i,
                                    "arm": arm,
                                    "target": name,
                                    "observed": int(keep.sum()),
                                    "estimate": estimate,
                                    "variance": variance,
                                    "truth": truth,
                                    "covered": int(abs(estimate - truth) <= half),
                                    "width": 2 * half,
                                    "squared_error": (estimate - truth) ** 2,
                                }
                            )
    data = pd.DataFrame(rows)
    data.to_csv(args.output / "predictions.csv", index=False)
    metrics = []
    for key, group in data.groupby(["world", "missingness", "seed", "arm", "target"]):
        low, high = proportion_confint(group.covered.sum(), len(group), method="wilson")
        metrics.append(
            dict(
                zip(["world", "missingness", "seed", "arm", "target"], key, strict=True)
            )
            | {
                "n": len(group),
                "coverage": group.covered.mean(),
                "wilson_low": low,
                "wilson_high": high,
                "mean_width": group.width.mean(),
                "mse": group.squared_error.mean(),
                "mean_observed": group.observed.mean(),
                "coverage_check": "PASS" if low <= 0.95 <= high else "FAIL",
            }
        )
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    summary = {
        "status": "KNOWN_PARAMETER_MISSINGNESS_REFERENCES_COMPLETE",
        "rows": len(rows),
        "metric_cells": len(metrics),
        "scope": protocol["scope"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

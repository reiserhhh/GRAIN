#!/usr/bin/env python3
"""Alternate full source-refit, Gaussian and Monte Carlo replay for F13 Delta."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def people(path):
    """Load numeric observations ordered by author and event time."""
    groups = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        groups.setdefault(r["author_id"], []).append(r)
    ids = sorted(groups)
    rows = [sorted(groups[k], key=lambda r: r["t"]) for k in ids]
    return (
        ids,
        np.array([[r["y"] for r in p] for p in rows]),
        np.array([r["z"] for r in rows[0]]),
    )


def conditional(mu, cov, noise, y, z):
    """Observation-space Gaussian conditioning independent of precision code."""
    x = np.column_stack([np.ones(len(z)), z])
    gain = cov @ x.T @ np.linalg.inv(x @ cov @ x.T + noise * np.eye(len(z)))
    return mu + (y - mu @ x.T) @ gain.T, cov - gain @ x @ cov


def wilson(success, n):
    """Wilson95% binomial interval for independent application authors."""
    z = 1.959963984540054
    p = success / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def balanced_reml_objective(sigma, error, information_inv, scatter, n, df, rss):
    """Twice negative restricted log-likelihood up to parameter-free constants."""
    c = sigma + error * information_inv
    return float(
        (n - 1) * np.linalg.slogdet(c)[1]
        + np.trace(np.linalg.solve(c, scatter))
        + df * np.log(error)
        + rss / error
    )


def main():
    """Replay all 600 resamples and24m draws; retain calibration uncertainty."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.input / "protocol_before_run.json").read_text())
    max_fit = max_gauss = max_draw = 0.0
    max_reml_gap = 0.0
    fallback_count = 0
    reml = config.get("source_estimator") == "reml"
    expected_predictions = {}
    max_mixture = 0.0
    refits = draw_count = 0
    floor_count = 0
    for seed in config["seeds"]:
        root = args.source / f"L_{seed}"
        _, source, z = people(root / "source.jsonl")
        _ids, app, za = people(root / "application.jsonl")
        app = app[:, :24]
        za = za[:24]
        fits = json.loads((args.input / f"fits_{seed}.json").read_text())
        with np.load(args.input / f"checkpoint_{seed}.npz") as stored:
            checkpoint = {k: stored[k] for k in stored.files}
        for b, fit in enumerate(fits):
            indices = np.random.default_rng(
                np.random.SeedSequence([seed, 13, 0, b, 0])
            ).integers(0, len(source), len(source))
            np.testing.assert_array_equal(indices, fit["source_indices"])
            y = source[indices]
            coef = np.column_stack([y.mean(axis=1), y @ z / (z @ z)])
            residual = y - coef[:, 0, None] - coef[:, 1, None] * z
            noise = np.sum(residual**2) / (len(y) * (len(z) - 2))
            mu = coef.mean(axis=0)
            centered = coef - mu
            raw = centered.T @ centered / (len(y) - 1) - noise * np.diag(
                [1 / len(z), 1 / (z @ z)]
            )
            eigen, vectors = np.linalg.eigh(raw)
            cov = (vectors * np.maximum(eigen, 1e-6)) @ vectors.T
            max_fit = max(
                max_fit,
                float(np.max(np.abs(mu - fit["source_mean"]))),
                float(np.max(np.abs(cov - fit["source_covariance"]))),
                abs(noise - fit["noise"]),
            )
            floor_count += int(np.any(eigen < 1e-6))
            if reml:
                # Balanced-design REML factors into OLS coefficient and residual laws.
                # Interior moment estimates give its exact optimum, not a second fit.
                if np.any(eigen <= 0):
                    raise ValueError("boundary fit requires constrained REML audit")
                information_inv = np.diag([1 / len(z), 1 / (z @ z)])
                scatter = centered.T @ centered

                objective_args = (
                    information_inv,
                    scatter,
                    len(y),
                    len(y) * (len(z) - 2),
                    np.sum(residual**2),
                )
                optimum = balanced_reml_objective(cov, noise, *objective_args)
                cov = np.asarray(fit["source_covariance"])
                noise = fit["noise"]
                mu = np.asarray(fit["source_mean"])
                gap = (
                    balanced_reml_objective(cov, noise, *objective_args) - optimum
                ) / len(y)
                max_reml_gap = max(max_reml_gap, abs(gap))
                if gap < -1e-8 or gap > 1e-3:
                    raise ValueError(f"REML objective gap per author {gap}")
                fallback_count += int(fit["estimator_diagnostics"]["fallback"])
            m, v = conditional(mu, cov, noise, app, za)
            v = (v + v.T) / 2
            max_gauss = max(
                max_gauss,
                float(np.max(np.abs(2 * m[:, 1] - checkpoint["means"][b]))),
                abs(4 * v[1, 1] - checkpoint["variances"][b]),
            )
            # Same recorded random stream, alternate explicit two-dimensional root.
            root00 = np.sqrt(v[0, 0])
            root10 = v[1, 0] / root00
            root11 = np.sqrt(v[1, 1] - root10**2)
            normal = np.random.default_rng(
                np.random.SeedSequence([seed, 13, 0, b, 1])
            ).normal(size=(config["inner"], len(app), 2))
            delta = 2 * (
                m[None, :, 1] + root10 * normal[:, :, 0] + root11 * normal[:, :, 1]
            )
            max_draw = max(
                max_draw,
                float(np.max(np.abs(delta.mean(axis=0) - checkpoint["draw_means"][b]))),
                float(
                    np.max(
                        np.abs(
                            np.mean(delta**2, axis=0) - checkpoint["draw_seconds"][b]
                        )
                    )
                ),
            )
            refits += 1
            draw_count += delta.size
        mixture_mean = checkpoint["means"].mean(axis=0)
        within = np.full(len(app), checkpoint["variances"].mean())
        between = checkpoint["means"].var(axis=0)
        mc_mean = checkpoint["draw_means"].mean(axis=0)
        mc_var = checkpoint["draw_seconds"].mean(axis=0) - mc_mean**2
        with np.load(args.input / f"mixture_{seed}.npz") as saved:
            for name, value in {
                "within": within,
                "between": between,
                "mean": mixture_mean,
                "mc_mean": mc_mean,
                "mc_var": mc_var,
            }.items():
                max_mixture = max(
                    max_mixture, float(np.max(np.abs(value - saved[name])))
                )
        if reml:
            plugin = json.loads((args.input / f"plugin_fit_{seed}.json").read_text())
            pmu, pcov, pnoise = (
                np.array(plugin["mu"]),
                np.array(plugin["Sigma"]),
                plugin["sigma2"],
            )
        else:
            beta = np.column_stack([source.mean(axis=1), source @ z / (z @ z)])
            pmu = beta.mean(axis=0)
            pnoise = np.sum((source - beta[:, 0, None] - beta[:, 1, None] * z) ** 2) / (
                len(source) * (len(z) - 2)
            )
            raw = np.cov(beta, rowvar=False) - pnoise * np.diag(
                [1 / len(z), 1 / (z @ z)]
            )
            eig, vectors = np.linalg.eigh(raw)
            pcov = (vectors * np.maximum(eig, 1e-6)) @ vectors.T
        pm, pv = conditional(pmu, pcov, pnoise, app, za)
        om, ov = conditional(np.array([0.0, 0.5]), np.diag([1.0, 0.25]), 0.25, app, za)
        target = {
            r["author_id"]: 2 * r["b"]
            for r in json.loads((root / "truth.json").read_text())
        }
        for name, mean, variance in (
            ("oracle", 2 * om[:, 1], np.full(len(app), 4 * ov[1, 1])),
            (
                "plugin_reml" if reml else "plugin_moment",
                2 * pm[:, 1],
                np.full(len(app), 4 * pv[1, 1]),
            ),
            ("source_mixture", mixture_mean, within + between),
            ("nested_mc", mc_mean, mc_var),
        ):
            for i, author in enumerate(_ids):
                expected_predictions[(str(seed), author, name)] = (
                    mean[i],
                    variance[i],
                    target[author],
                )
    with (args.input / "predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    coverage = []
    max_metric = 0.0
    seen_predictions = set()
    for row in predictions:
        key = (row["seed"], row["author_id"], row["method"])
        if key in seen_predictions:
            raise ValueError("duplicate prediction")
        seen_predictions.add(key)
        expected = expected_predictions[key]
        max_mixture = max(
            max_mixture,
            max(
                abs(float(row[k]) - v)
                for k, v in zip(("mean", "variance", "truth"), expected, strict=True)
            ),
        )
    if seen_predictions != set(expected_predictions):
        raise ValueError("missing prediction rows")
    with (args.input / "metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            group = [
                r
                for r in predictions
                if r["seed"] == row["seed"] and r["method"] == row["method"]
            ]
            p = np.array([float(r["mean"]) for r in group])
            v = np.array([float(r["variance"]) for r in group])
            truth = np.array([float(r["truth"]) for r in group])
            ok = np.abs(p - truth) <= 1.96 * np.sqrt(v)
            for flag, r in zip(ok, group, strict=True):
                assert flag == (r["covered"] == "True")
            values = {
                "mse": np.mean((p - truth) ** 2),
                "coverage": ok.mean(),
                "mean_width": np.mean(3.92 * np.sqrt(v)),
                "mean_variance": v.mean(),
            }
            max_metric = max(
                max_metric,
                max(abs(value - float(row[k])) for k, value in values.items()),
            )
            low, high = wilson(int(ok.sum()), len(ok))
            coverage.append(
                {
                    "seed": row["seed"],
                    "method": row["method"],
                    "n": len(ok),
                    "covered": int(ok.sum()),
                    "rate": float(ok.mean()),
                    "wilson_low": low,
                    "wilson_high": high,
                    "contains_nominal": bool(low <= 0.95 <= high),
                }
            )
    if max(max_gauss, max_draw, max_metric, max_mixture, 0 if reml else max_fit) > 1e-9:
        raise ValueError("nested replay mismatch")
    with (args.output / "coverage_wilson.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    result = {
        "status": "ALL_SOURCE_REFITS_AND_CONDITIONAL_DRAWS_REPLAYED",
        "source_refits": refits,
        "conditional_draws": draw_count,
        "floor_count": floor_count,
        "max_source_error": max_fit,
        "source_comparison": "exact balanced interior REML optimum"
        if reml
        else "alternate moments",
        "max_reml_objective_gap_per_author": max_reml_gap,
        "fallback_count": fallback_count,
        "max_gaussian_error": max_gauss,
        "max_draw_moment_error": max_draw,
        "max_metric_error": max_metric,
        "max_final_prediction_and_mixture_error": max_mixture,
        "prediction_rows_checked": len(seen_predictions),
        "coverage_cells": len(coverage),
        "cells_containing_nominal": sum(r["contains_nominal"] for r in coverage),
        "scope": "per-seed author Wilson intervals conditional on shared source fit; not proof of universal calibration",
        "independent_agent": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

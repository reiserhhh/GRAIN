#!/usr/bin/env python3
"""Actual 200-source-bootstrap x200 conditional draws for F04 linear Delta."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_people(path, role):
    """Group existing records, rejecting duplicate event IDs and role mixing."""
    people = {}
    seen = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["role"] != role or r["event_id"] in seen:
            raise ValueError("role/duplicate violation")
        seen.add(r["event_id"])
        people.setdefault(r["author_id"], []).append(r)
    ids = sorted(people)
    sequences = [sorted(people[i], key=lambda r: r["t"]) for i in ids]
    conditions = np.array([[r["z"] for r in p] for p in sequences])
    if not np.all(conditions == conditions[0]):
        raise ValueError("balanced L panel required")
    return (
        ids,
        np.array([[r["y"] for r in p] for p in sequences]),
        np.column_stack([np.ones(conditions.shape[1]), conditions[0]]),
    )


def fit_moments(y, x):
    """Source noise-corrected moment arm, explicitly not a REML refit."""
    beta = np.linalg.lstsq(x, y.T, rcond=None)[0].T
    residual = y - beta @ x.T
    noise = float(np.sum(residual**2) / (len(y) * (len(x) - 2)))
    raw = np.cov(beta, rowvar=False) - noise * np.linalg.inv(x.T @ x)
    eigen, vectors = np.linalg.eigh(raw)
    covariance = (vectors * np.maximum(eigen, 1e-6)) @ vectors.T
    return beta.mean(axis=0), covariance, noise, eigen


def posterior(mu, covariance, noise, y, x):
    """Gaussian conditional parameters using two-dimensional precision algebra."""
    precision = np.linalg.inv(covariance)
    v = np.linalg.inv(precision + x.T @ x / noise)
    means = (mu @ precision + y @ x / noise) @ v
    return means, v


def fit_reml(y, x):
    """Run the frozen MixedLM procedure with distinct bootstrap-copy groups."""
    from grain_research.gptpro_foundation_v3.f04_recovery import fit_source_lmm

    rows = [
        {
            "role": "source",
            "author_id": f"bootstrap_copy_{i}",
            "event_id": f"bootstrap_copy_{i}/{t}",
            "z": float(x[t, 1]),
            "y": float(value),
        }
        for i, person in enumerate(y)
        for t, value in enumerate(person)
    ]
    fit = fit_source_lmm(rows)
    cov = np.asarray(fit["Sigma"])
    return (np.asarray(fit["mu"]), cov, fit["sigma2"], np.linalg.eigvalsh(cov), fit)


def main():
    """Checkpoint every ten source resamples and retain Monte Carlo moments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--estimator", choices=("moment", "reml"), default="moment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    outer, inner = (2, 10) if args.smoke else (200, 200)
    config = {
        "seeds": seeds,
        "outer": outer,
        "inner": inner,
        "prefix": 24,
        "world": "F04 L complete",
        "source_estimator": args.estimator,
        "optimizer_contract": "REML: lbfgs200, then powell200, labelled moment fallback; bootstrap copies are distinct random-effect groups",
        "target": "Delta=2b",
        "scope": "conditional Gaussian mixture induced by empirical source bootstrap,not exact hierarchical posterior",
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    metrics = []
    results = []
    total_draws = 0
    for seed in seeds:
        directory = args.source / f"L_{seed}"
        source_ids, source, x = load_people(directory / "source.jsonl", "source")
        ids, app, xa = load_people(directory / "application.jsonl", "application")
        if set(source_ids) & set(ids):
            raise ValueError("source/application author overlap")
        if args.smoke:
            source = source[:12]
            ids = ids[:6]
            app = app[:6]
        y = app[:, :24]
        design = xa[:24]
        target_index = {
            r["author_id"]: r
            for r in json.loads((directory / "truth.json").read_text())
        }
        truth = np.array([2 * target_index[i]["b"] for i in ids])
        estimator = fit_reml if args.estimator == "reml" else fit_moments
        source_fit = estimator(source, x)
        if args.estimator == "reml":
            (args.output / f"plugin_fit_{seed}.json").write_text(
                json.dumps(source_fit[4], indent=2) + "\n"
            )
        plug, plug_v = posterior(*source_fit[:3], y, design)
        oracle, oracle_v = posterior(
            np.array([0.0, 0.5]), np.diag([1.0, 0.25]), 0.25, y, design
        )
        means = []
        variances = []
        draw_means = []
        draw_seconds = []
        fits = []
        for b in range(outer):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 13, 0, b, 0]))
            selected = rng.integers(0, len(source), len(source))
            fitted = estimator(source[selected], x)
            mu, cov, noise, eigen = fitted[:4]
            m, v = posterior(mu, cov, noise, y, design)
            draws = (
                np.random.default_rng(
                    np.random.SeedSequence([seed, 13, 0, b, 1])
                ).normal(size=(inner, len(y), 2))
                @ np.linalg.cholesky(v).T
                + m
            )
            delta = 2 * draws[:, :, 1]
            means.append(2 * m[:, 1])
            variances.append(4 * v[1, 1])
            draw_means.append(delta.mean(axis=0))
            draw_seconds.append((delta**2).mean(axis=0))
            fits.append(
                {
                    "bootstrap": b,
                    "source_mean": mu.tolist(),
                    "source_covariance": cov.tolist(),
                    "noise": noise,
                    "raw_eigenvalues": eigen.tolist(),
                    "floored": bool(np.any(eigen < 1e-6)),
                    "source_indices": selected.tolist(),
                    "estimator_diagnostics": fitted[4] if len(fitted) > 4 else None,
                }
            )
            total_draws += inner * len(y)
            if (b + 1) % 10 == 0 or b + 1 == outer:
                np.savez(
                    args.output / f"checkpoint_{seed}.npz",
                    means=means,
                    variances=variances,
                    draw_means=draw_means,
                    draw_seconds=draw_seconds,
                )
                (args.output / f"fits_{seed}.json").write_text(json.dumps(fits) + "\n")
                (args.output / "checkpoint.json").write_text(
                    json.dumps(
                        {"seed": seed, "outer_completed": b + 1, "outer_total": outer}
                    )
                    + "\n"
                )
        means = np.array(means)
        variances = np.array(variances)
        center = means.mean(axis=0)
        within = np.full(len(y), variances.mean())
        between = means.var(axis=0)
        mc_mean = np.mean(draw_means, axis=0)
        mc_var = np.mean(draw_seconds, axis=0) - mc_mean**2
        np.savez(
            args.output / f"mixture_{seed}.npz",
            within=within,
            between=between,
            mean=center,
            mc_mean=mc_mean,
            mc_var=mc_var,
        )
        for name, m, v in (
            ("oracle", 2 * oracle[:, 1], np.full(len(y), 4 * oracle_v[1, 1])),
            (
                f"plugin_{args.estimator}",
                2 * plug[:, 1],
                np.full(len(y), 4 * plug_v[1, 1]),
            ),
            ("source_mixture", center, within + between),
            ("nested_mc", mc_mean, mc_var),
        ):
            covered = np.abs(m - truth) <= 1.96 * np.sqrt(v)
            for i in range(len(y)):
                results.append(
                    {
                        "seed": seed,
                        "author_id": ids[i],
                        "method": name,
                        "mean": m[i],
                        "variance": v[i],
                        "truth": truth[i],
                        "covered": bool(covered[i]),
                    }
                )
            metrics.append(
                {
                    "seed": seed,
                    "method": name,
                    "n": len(y),
                    "mse": float(np.mean((m - truth) ** 2)),
                    "coverage": float(covered.mean()),
                    "mean_width": float(np.mean(3.92 * np.sqrt(v))),
                    "mean_variance": float(v.mean()),
                }
            )
    for filename, rows in (("predictions.csv", results), ("metrics.csv", metrics)):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "status": f"NESTED_LINEAR_DELTA_{args.estimator.upper()}_ARM_COMPUTED_AUDIT_PENDING",
        "source_fits": outer * len(seeds),
        "conditional_person_draws": total_draws,
        "outer": outer,
        "inner": inner,
        "remaining": "F05/F08 targets,missingness/correlation/history/shift worlds,independent audit and contracts",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

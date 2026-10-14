#!/usr/bin/env python3
"""Actual author bootstrap source fits for missingness and AR references."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f13_source_covariance import (
    fit_source_covariance,
)


def main():
    """Learn covariance magnitudes, holding residual-kernel shape explicitly fixed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    n, events, outer = (12, 8, 2) if args.smoke else (300, 32, 200)
    protocol = {
        "seeds": seeds,
        "authors": n,
        "events": events,
        "bootstrap": outer,
        "source_rng": "SeedSequence([seed,13,4,0])",
        "fit": "unweighted covariance-basis least squares and source-mean regression; no eigenvalue clipping",
        "worlds": ["iid", "ar06"],
        "assumed_rho": [0.0, 0.6],
        "boundary": "rho is fixed as a declared kernel, NOT estimated; source complete, application masks applied later",
        "files": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in (
                Path(__file__),
                Path("grain_research/gptpro_foundation_v3/f13_source_covariance.py"),
            )
        ],
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], events // 4)
    x = np.column_stack([np.ones(events), z])
    lag = abs(np.arange(events)[:, None] - np.arange(events))
    completed = failed = 0
    for seed in seeds:
        ps, es = np.random.SeedSequence([seed, 13, 4, 0]).spawn(2)
        theta = np.random.default_rng(ps).normal(size=(n, 2)) * [1.0, 0.5] + [0.0, 0.5]
        noise = np.random.default_rng(es).normal(size=(n, events))
        for world, rho in (("iid", 0.0), ("ar06", 0.6)):
            y = theta @ x.T + 0.5 * noise @ np.linalg.cholesky(rho**lag).T
            np.savez_compressed(
                args.output / f"source_{world}_{seed}.npz", values=y, x=x
            )
            for assumed in (0.0, 0.6):
                c = assumed**lag
                root = args.output / f"{world}_{seed}_assumed{assumed}"
                root.mkdir()
                (root / "plugin.json").write_text(
                    json.dumps(fit_source_covariance(y, x, c)) + "\n"
                )
                for b in range(outer):
                    recipe = [seed, 13, 4, b, 1]
                    indices = np.random.default_rng(
                        np.random.SeedSequence(recipe)
                    ).integers(0, n, n)
                    fit = fit_source_covariance(y[indices], x, c)
                    # Independent reduced normal equations replay the four bases.
                    empirical = np.cov(y[indices], rowvar=False)
                    xx = [
                        np.outer(x[:, 0], x[:, 0]),
                        np.outer(x[:, 0], x[:, 1]) + np.outer(x[:, 1], x[:, 0]),
                        np.outer(x[:, 1], x[:, 1]),
                        c,
                    ]
                    gram = np.array([[np.sum(a * d) for d in xx] for a in xx])
                    rhs = np.array([np.sum(a * empirical) for a in xx])
                    np.testing.assert_allclose(
                        np.linalg.solve(gram, rhs),
                        fit["coefficients"],
                        atol=1e-10,
                        rtol=1e-8,
                    )
                    payload = {
                        "world": world,
                        "seed": seed,
                        "assumed_rho": assumed,
                        "bootstrap": b,
                        "source_indices": indices.tolist(),
                        "rng_recipe": recipe,
                        "fit": fit,
                    }
                    (root / f"fit_{b:03d}.json").write_text(json.dumps(payload) + "\n")
                    completed += 1
                    failed += fit["status"] != "SUPPORTED"
                    if completed % 10 == 0:
                        (args.output / "checkpoint.json").write_text(
                            json.dumps(
                                {
                                    "completed": completed,
                                    "invalid_variance_fits": failed,
                                }
                            )
                            + "\n"
                        )
    summary = {
        "status": "SOURCE_MOMENT_BOOTSTRAP_COMPLETE",
        "fits": completed,
        "invalid_variance_fits": failed,
        "application": "NOT_YET_SCORED",
        "scope": protocol["boundary"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

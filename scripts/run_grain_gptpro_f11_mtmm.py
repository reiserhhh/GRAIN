#!/usr/bin/env python3
"""Execute source-fitted MTMM worlds and leave application truth out of fitting."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.gptpro_foundation_v3.f11_mtmm import (
    designs,
    fit_mtmm,
    moment_fit,
    predict_traits,
)


def main():
    """Run three seeds/worlds with matched source-only moment and REML comparators."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "seeds": [20260915, 20260916, 20260917],
        "worlds": ["base", "shared_event", "conditional_loading"],
        "counts": [300, 100, 200],
        "source_fit": "18 fixed cell means; REML covariance; no application target access",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sigma = np.full((3, 3), 0.3)
    np.fill_diagonal(sigma, 1.0)
    fits, metrics = [], []
    for seed in config["seeds"]:
        for world_index, world in enumerate(config["worlds"]):
            conditional, event = (
                world == "conditional_loading",
                0.0 if world == "base" else 0.25,
            )
            f, m, c = designs(conditional)
            data = {}
            for role_index, (role, n) in enumerate(
                zip(
                    ("source", "validation", "application"),
                    config["counts"],
                    strict=True,
                )
            ):
                streams = np.random.SeedSequence(
                    [seed, 11, world_index, role_index]
                ).spawn(4)
                trait = np.random.default_rng(streams[0]).multivariate_normal(
                    np.zeros(3), sigma, size=n
                )
                method = np.random.default_rng(streams[1]).normal(
                    scale=0.5, size=(n, 3)
                )
                shared = np.random.default_rng(streams[2]).normal(
                    scale=np.sqrt(event), size=(n, 2)
                )
                error = np.random.default_rng(streams[3]).normal(
                    scale=0.5, size=(n, 18)
                )
                observed = trait @ f.T + method @ m.T + shared @ c.T + error
                data[role] = observed, trait
                np.savez(
                    args.output / f"data_{seed}_{world}_{role}.npz", observed=observed
                )
                np.savez(
                    args.output / f"truth_{seed}_{world}_{role}.npz",
                    trait=trait,
                    method=method,
                    shared=shared,
                    error=error,
                )
            models = {
                "moment": moment_fit(data["source"][0], conditional, event > 0),
                "reml": fit_mtmm(data["source"][0], conditional, event > 0),
            }
            # Oracle is a declared diagnostic, never an alternative source fit.
            models["oracle"] = {
                "sigma": sigma,
                "method": 0.25,
                "event": event,
                "error": 0.25,
                "mean": np.zeros(18),
            }
            for name, model in models.items():
                serial = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in model.items()
                }
                (args.output / f"fit_{seed}_{world}_{name}.json").write_text(
                    json.dumps(serial, indent=2) + "\n"
                )
                fits.append(
                    {
                        "seed": seed,
                        "world": world,
                        "method": name,
                        "trait_cov_rmse": float(
                            np.sqrt(np.mean((model["sigma"] - sigma) ** 2))
                        ),
                        "reader_variance": model["method"],
                        "event_variance": model["event"],
                        "error_variance": model["error"],
                        "converged": model.get("success"),
                        "gradient_max": model.get("gradient_max"),
                    }
                )
                for role in ("validation", "application"):
                    y, target = data[role]
                    for panel, cells in (
                        ("all_two_occasions", None),
                        ("same_event_three_readers", np.arange(0, 18, 2)),
                        ("one_reader_two_occasions", np.array([0, 1, 6, 7, 12, 13])),
                    ):
                        prediction, variance = predict_traits(
                            model, y, conditional, cells
                        )
                        np.savez(
                            args.output
                            / f"pred_{seed}_{world}_{name}_{role}_{panel}.npz",
                            prediction=prediction,
                            variance=variance,
                        )
                        se = np.sqrt(np.diag(variance))
                        metrics.append(
                            {
                                "seed": seed,
                                "world": world,
                                "method": name,
                                "role": role,
                                "panel": panel,
                                "n": len(y),
                                "mse": float(np.mean((prediction - target) ** 2)),
                                "coverage": float(
                                    np.mean(np.abs(prediction - target) <= 1.96 * se)
                                ),
                                "mean_variance": float(np.mean(se**2)),
                            }
                        )
    for filename, rows in (("fits.csv", fits), ("metrics.csv", metrics)):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    reml = [r for r in fits if r["method"] == "reml"]
    summary = {
        "status": "MTMM_COMPUTED_AUDIT_PENDING",
        "authors": 5400,
        "reml_fits": len(reml),
        "converged": sum(r["converged"] for r in reml),
        "max_gradient": max(r["gradient_max"] for r in reml),
        "metric_rows": len(metrics),
        "note": "panels have different cell budgets, not equal-cost comparison; plug-in intervals omit source-fit uncertainty",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

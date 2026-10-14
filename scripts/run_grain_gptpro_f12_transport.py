#!/usr/bin/env python3
"""Execute pilot-only transport under population shifts and four reader channels."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.gptpro_foundation_v3.f12_transport import (
    fit_transport,
    transport_score,
)


def transform(values, z, world):
    """Apply one frozen channel change, not a change of author parameters."""
    if world == "identity":
        return values.copy()
    if world == "global":
        return 1 + 1.5 * values
    if world == "conditional":
        return z + (1 + 0.25 * z) * values
    return values + 0.1 * values**3


def write_csv(path, rows):
    """Write deterministic tables, preserving missing values as empty fields."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Fit 60 pilot authors, check40, and apply to200 without model selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    counts = [12, 10, 6] if args.smoke else [300, 100, 200]
    fit_n = 6 if args.smoke else 60
    worlds = ["identity", "global", "conditional", "nonlinear"]
    protocol = {
        "seeds": seeds,
        "source_pilot_application": counts,
        "pilot_fit": fit_n,
        "pilot_validation": counts[1] - fit_n,
        "worlds": worlds,
        "events": 32,
        "scoring_prefix": 24,
        "old_population_mean": [0, 0.5],
        "old_covariance": [1, 0.25],
        "new_population_mean": [1, 0.2],
        "new_covariance": [1.4, 0.7],
        "paired_material_noise": "same old observation transformed deterministically; no independent reader noise",
        "interval": "personal OLS using source-estimated error, plug-in calibration uncertainty omitted",
        "smoke": args.smoke,
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], 8)
    x = np.column_stack([np.ones(32), z])
    target_x = np.column_stack([np.ones(4), z[:4]])
    rows, summary_rows, model_rows = [], [], []
    for seed in seeds:
        data = {}
        for role_id, (role, n) in enumerate(
            zip(("source", "pilot", "application"), counts, strict=True)
        ):
            parameter_seed, error_seed = np.random.SeedSequence(
                [seed, 12, 0, role_id]
            ).spawn(2)
            rng = np.random.default_rng(parameter_seed)
            mean = np.array([0.0, 0.5]) if role == "source" else np.array([1.0, 0.2])
            variance = (
                np.array([1.0, 0.25]) if role == "source" else np.array([1.4, 0.7])
            )
            theta = rng.normal(size=(n, 2)) * np.sqrt(variance) + mean
            error = np.random.default_rng(error_seed).normal(scale=0.5, size=(n, 32))
            observed = theta @ x.T + error
            data[role] = observed, theta
            np.savez(
                args.output / f"observations_{seed}_{role}.npz",
                old=observed,
                conditions=z,
            )
            np.savez(args.output / f"truth_{seed}_{role}.npz", theta=theta, error=error)
        source = data["source"][0]
        estimates = np.linalg.lstsq(x, source.T, rcond=None)[0].T
        residual = source - estimates @ x.T
        error_variance = float(np.sum(residual**2) / (len(source) * 30))
        (args.output / f"source_reference_{seed}.json").write_text(
            json.dumps(
                {
                    "parameter_mean": estimates.mean(axis=0).tolist(),
                    "estimated_error_variance": error_variance,
                    "note": "reference units retained, not renormalized to new population",
                },
                indent=2,
            )
            + "\n"
        )
        for world in worlds:
            pilot = data["pilot"][0]
            new_pilot = transform(pilot, z, world)
            models = {
                mode: fit_transport(
                    pilot[:fit_n].ravel(),
                    new_pilot[:fit_n].ravel(),
                    np.tile(z, fit_n),
                    mode,
                )
                for mode in ("global", "conditional")
            }
            if world == "nonlinear":
                models["cubic"] = fit_transport(
                    pilot[:fit_n].ravel(),
                    new_pilot[:fit_n].ravel(),
                    np.tile(z, fit_n),
                    "cubic",
                )
            # Missing-condition certificate uses only one condition; no imputation.
            mask = np.tile(z, fit_n) == -1
            missing = fit_transport(
                pilot[:fit_n].ravel()[mask],
                new_pilot[:fit_n].ravel()[mask],
                np.tile(z, fit_n)[mask],
                "conditional",
            )
            if missing["status"] != "MISSING_ANCHOR":
                raise ValueError("rank deficiency not detected")
            for name, model in {
                **models,
                "missing_condition_certificate": missing,
            }.items():
                serial = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in model.items()
                }
                (args.output / f"calibration_{seed}_{world}_{name}.json").write_text(
                    json.dumps(serial, indent=2) + "\n"
                )
                model_rows.append(
                    {
                        "seed": seed,
                        "world": world,
                        "method": name,
                        "status": model["status"],
                        "rank": model["rank"],
                        "min_derivative": model.get("min_derivative_on_pilot_range"),
                    }
                )
            for role in ("pilot_validation", "application"):
                observed, theta = (
                    (data["pilot"][0][fit_n:], data["pilot"][1][fit_n:])
                    if role == "pilot_validation"
                    else data["application"]
                )
                shifted = transform(observed, z, world)
                np.savez(args.output / f"new_{seed}_{world}_{role}.npz", new=shifted)
                for method in ["uncalibrated", *models]:
                    score, support = (
                        (shifted, np.ones_like(shifted, dtype=bool))
                        if method == "uncalibrated"
                        else transport_score(
                            models[method], shifted, np.broadcast_to(z, shifted.shape)
                        )
                    )
                    np.savez(
                        args.output / f"scores_{seed}_{world}_{role}_{method}.npz",
                        score=score,
                        support=support,
                    )
                    valid = np.all(support[:, :24], axis=1)
                    predicted_delta, true_delta = [], []
                    mse_cells, coverage = [], []
                    for i in range(len(score)):
                        row = {
                            "seed": seed,
                            "world": world,
                            "role": role,
                            "method": method,
                            "index": i,
                            "supported": bool(valid[i]),
                            "cell_mse": None,
                            "delta_prediction": None,
                            "delta_truth": float(2 * theta[i, 1]),
                            "delta_variance": None,
                            "delta_covered": None,
                        }
                        if valid[i]:
                            beta = np.linalg.lstsq(x[:24], score[i, :24], rcond=None)[0]
                            cell_error = float(
                                np.mean((target_x @ beta - target_x @ theta[i]) ** 2)
                            )
                            variance = float(
                                4
                                * error_variance
                                * np.linalg.inv(x[:24].T @ x[:24])[1, 1]
                            )
                            delta = float(2 * beta[1])
                            target = float(2 * theta[i, 1])
                            covered = abs(delta - target) <= 1.96 * np.sqrt(variance)
                            row.update(
                                cell_mse=cell_error,
                                delta_prediction=delta,
                                delta_variance=variance,
                                delta_covered=bool(covered),
                            )
                            predicted_delta.append(delta)
                            true_delta.append(target)
                            mse_cells.append(cell_error)
                            coverage.append(covered)
                        rows.append(row)
                    error = (score - observed)[support]
                    summary_rows.append(
                        {
                            "seed": seed,
                            "world": world,
                            "role": role,
                            "method": method,
                            "n": len(score),
                            "supported_authors": int(valid.sum()),
                            "supported_readings": int(support.sum()),
                            "material_mse": float(np.mean(error**2))
                            if len(error)
                            else None,
                            "cell_mse": float(np.mean(mse_cells))
                            if mse_cells
                            else None,
                            "delta_mse": float(
                                np.mean((np.array(predicted_delta) - true_delta) ** 2)
                            )
                            if predicted_delta
                            else None,
                            "delta_spearman": float(
                                spearmanr(predicted_delta, true_delta).statistic
                            )
                            if len(predicted_delta) > 2
                            else None,
                            "delta_coverage": float(np.mean(coverage))
                            if coverage
                            else None,
                        }
                    )
    write_csv(args.output / "author_metrics.csv", rows)
    write_csv(args.output / "metrics.csv", summary_rows)
    write_csv(args.output / "calibration_status.csv", model_rows)
    summary = {
        "status": "FOUR_CHANNEL_TRANSPORT_COMPUTED_AUDIT_PENDING",
        "seeds": seeds,
        "author_metric_rows": len(rows),
        "metric_rows": len(summary_rows),
        "all_models_source_pilot_only": True,
        "application_selection": False,
        "remaining": "signature identity,paired uncertainty,numerical replay and card contracts",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

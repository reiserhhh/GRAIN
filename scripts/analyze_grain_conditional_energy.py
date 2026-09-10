#!/usr/bin/env python3
"""Evaluate frozen first/second moments, keeping strength and signed score apart."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.comment_carrier import score
from grain_research.vector_mechanisms.conditional_energy import (
    energy_gap,
    score_mixture,
)
from scripts.run_grain_conditional_energy import PARENT, verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def corr(x, y):
    """Scale-invariant Pearson for tiny numerical actions; constants give NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("Matched finite scalar action rows required")
    x, y = x - x.mean(), y - y.mean()
    sx, sy = np.max(abs(x)), np.max(abs(y))
    if sx == 0 or sy == 0:
        return float("nan")
    x, y = x / sx, y / sy
    return float(np.clip(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1))


def within(x, groups):
    """Remove fold means for interpretation only, never create a refitted output."""
    x, groups = np.asarray(x, float), np.asarray(groups)
    out = x.copy()
    for group in np.unique(groups):
        ix = groups == group
        if np.ptp(x[ix]) == 0:
            out[ix] = 0
        else:
            out[ix] -= x[ix].mean()
    return out


def author_draws(authors):
    """Keep repeated pair observations attached to the author bootstrap unit."""
    authors = np.asarray(authors)
    units = np.unique(authors)
    slots = [np.flatnonzero(authors == u) for u in units]
    rng = np.random.default_rng(2501)
    return [
        np.concatenate([slots[j] for j in draw])
        for draw in rng.integers(0, len(units), size=(2000, len(units)))
    ]


def ci(values):
    """Return a pointwise interval and the number of finite bootstrap replicates."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    return *np.quantile(values, [0.025, 0.975]), len(values)


def analyze(out, name):
    """Read model targets only after fixed moment construction, never human Y."""
    verified(out)
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "analyzer_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "human_labels_read": False,
            "new_neural_calls": 0,
        },
    )
    predictions = pd.read_csv(
        out / "private_moment_predictions.csv", dtype={"author_id": str}
    )
    with np.load(out / "private_moments.npz") as z:
        arrays = dict(z)
    models = read(out / "private_distribution_models.json")
    records = read(PARENT / "private_joint_predictions.json")
    lookup = {(r["author_id"], r["pair"], r["arm"]): r for r in records}
    rows = []
    for r in predictions.itertuples():
        old = lookup[r.author_id, r.pair, r.arm]
        model = models[r.model_index]
        if (
            old["fold"] != r.fold
            or model["fold"] != r.fold
            or model["pair"] != r.pair
            or model["arm"] != r.arm
            or r.author_id in model["training_authors"]
        ):
            raise ValueError("Author/role/input-family leakage or identity mismatch")
        j = r.row_index
        y, p = np.asarray(old["actual_mixed"]), np.asarray(old["predicted_mixed"])
        m, covariance, axis = (
            arrays["mean"][j],
            arrays["covariance"][j],
            arrays["axis"][j],
        )
        gap = energy_gap(y[None], p[None])
        score_account = score_mixture(
            old["z_add"], old["values"], arrays["weights"][j], model["targets"]
        )
        base = score(old["z_add"], old["values"])
        rows.append(
            {
                **r._asdict(),
                "actual_energy": float(y @ y),
                "actual_parallel_energy": float((y @ axis) ** 2),
                "actual_perpendicular_energy": float(y @ y - (y @ axis) ** 2),
                "moment_identity_error": abs(
                    r.energy - float(m @ m) - float(np.trace(covariance))
                ),
                "axis_identity_error": abs(
                    r.energy - r.parallel_energy - r.perpendicular_energy
                ),
                "minimum_covariance_eigenvalue": float(
                    np.linalg.eigvalsh(covariance).min()
                ),
                **{k: float(v[0]) for k, v in gap.items()},
                "actual_score_action": old["actual_score"] - base,
                "ridge_score_action": old["oracle_singletons_reconstruction"] - base,
                "mixture_mean_action": score_account["score_at_mean"] - base,
                "mixture_score_action": score_account["mean_score"] - base,
                **score_account,
            }
        )
    data = pd.DataFrame(rows).drop(columns="Index")
    data.to_csv(dest / "private_energy_accounts.csv", index=False)
    energy_rows, contrasts, geometry, link_rows = [], [], [], []
    for (arm, pair), g in data.groupby(["arm", "pair"]):
        g = g.sort_values("author_id")
        draws = author_draws(g.author_id)
        actual = g.actual_energy.to_numpy()
        for field in (
            "energy",
            "mean_energy",
            "old_vector_energy",
            "pair_mean_energy",
            "pair_energy_mean",
            "parallel_energy",
            "perpendicular_energy",
        ):
            target = g[
                "actual_parallel_energy"
                if field == "parallel_energy"
                else "actual_perpendicular_energy"
                if field == "perpendicular_energy"
                else "actual_energy"
            ].to_numpy()
            p = g[field].to_numpy()
            error = (target - p) ** 2
            low, high, finite = ci([corr(target[ix], p[ix]) for ix in draws])
            energy_rows.append(
                {
                    "arm": arm,
                    "pair": pair,
                    "statistic": field,
                    "authors": len(g),
                    "r": corr(target, p),
                    "r_low": low,
                    "r_high": high,
                    "finite_bootstraps": finite,
                    "within_fold_r": corr(within(target, g.fold), within(p, g.fold)),
                    "mse_over_zero": float(error.sum() / (target @ target)),
                    "mae": float(np.mean(abs(target - p))),
                    "predicted_mean": float(p.mean()),
                    "actual_mean": float(target.mean()),
                }
            )
        for baseline in (
            "mean_energy",
            "old_vector_energy",
            "pair_mean_energy",
            "pair_energy_mean",
        ):
            improvement = (actual - g[baseline].to_numpy()) ** 2 - (
                actual - g.energy.to_numpy()
            ) ** 2
            low, high, finite = ci([improvement[ix].mean() for ix in draws])
            contrasts.append(
                {
                    "arm": arm,
                    "pair": pair,
                    "baseline": baseline,
                    "mse_improvement": float(improvement.mean()),
                    "low": low,
                    "high": high,
                    "finite_bootstraps": finite,
                }
            )
        geometry.append(
            {
                "arm": arm,
                "pair": pair,
                "authors": len(g),
                "actual_energy_mean": float(actual.mean()),
                "ridge_energy_mean": float(g.old_vector_energy.mean()),
                "ridge_gap_mean": float(g.energy_gap.mean()),
                "ridge_residual_energy_mean": float(g.residual_energy.mean()),
                "ridge_cross_term_mean": float(g.cross_term.mean()),
                "mixture_energy_mean": float(g.energy.mean()),
                "mixture_mean_energy_mean": float(g.mean_energy.mean()),
                "mixture_variance_trace_mean": float(g.variance_trace.mean()),
                "reference_mass_mean": float(g.reference_mass.mean()),
                "effective_atoms_mean": float(g.effective_atoms.mean()),
            }
        )
        y = g.actual_score_action.to_numpy()
        for field in (
            "ridge_score_action",
            "mixture_mean_action",
            "mixture_score_action",
        ):
            p = g[field].to_numpy()
            link_rows.append(
                {
                    "arm": arm,
                    "pair": pair,
                    "reader": field,
                    "authors": len(g),
                    "r": corr(y, p),
                    "mse_over_zero": float(np.sum((y - p) ** 2) / (y @ y)),
                    "actual_rms": float(np.sqrt(np.mean(y**2))),
                    "jensen_difference_rms": float(
                        np.sqrt(np.mean(g.mixture_minus_mean_score**2))
                    ),
                    "covariance_correction_rms": float(
                        np.sqrt(np.mean(g.covariance_correction**2))
                    ),
                    "higher_order_rms": float(
                        np.sqrt(np.mean(g.higher_order_remainder**2))
                    ),
                    "oracle_singletons_required": True,
                }
            )
    for file, values in (
        ("energy_comparison.csv", energy_rows),
        ("energy_paired_mse.csv", contrasts),
        ("moment_geometry.csv", geometry),
        ("score_link_comparison.csv", link_rows),
    ):
        pd.DataFrame(values).to_csv(dest / file, index=False)
    summary = {
        "status": "COMPLETE",
        "authors": 24,
        "pairs": 48,
        "input_families": 2,
        "new_neural_calls": 0,
        "human_labels_read": False,
        "energy_gap_identity_max": float(data.identity_error.max()),
        "moment_identity_max": float(data.moment_identity_error.max()),
        "axis_identity_max": float(data.axis_identity_error.max()),
        "minimum_covariance_eigenvalue": float(
            data.minimum_covariance_eigenvalue.min()
        ),
        "all_scoring_origins_oracle_conditioned": True,
        "files": {
            name: sha(dest / name)
            for name in (
                "private_energy_accounts.csv",
                "energy_comparison.csv",
                "energy_paired_mse.csv",
                "moment_geometry.csv",
                "score_link_comparison.csv",
            )
        },
    }
    write(dest / "summary.json", summary)
    table = pd.DataFrame(energy_rows)
    print(
        table[
            table.statistic.isin(
                ["energy", "mean_energy", "old_vector_energy", "pair_energy_mean"]
            )
        ][
            ["arm", "pair", "statistic", "r", "within_fold_r", "mse_over_zero"]
        ].to_string(index=False)
    )
    print({k: v for k, v in summary.items() if k != "files"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="logic_conditional_energy_v1")
    parser.add_argument("--name", default="analysis")
    args = parser.parse_args()
    if any(not n.replace("_", "").isalnum() for n in (args.output_name, args.name)):
        raise ValueError("Simple output names required")
    analyze(BASE / args.output_name, args.name)

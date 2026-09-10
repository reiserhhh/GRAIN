#!/usr/bin/env python3
"""Evaluate frozen NEXT35 residual input arms and expose source/held geometry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.content_coefficients import coefficient_field
from grain_research.vector_mechanisms.native_material_reader import direction
from grain_research.vector_mechanisms.position_reader import contractions, errors
from scripts.analyze_grain_position_reader import action_metrics
from scripts.run_grain_query_residual_input import ARMS, NATIVE, SOURCE, verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def error_parts(j, predicted_coefficients, basis, directions):
    """Exact orthogonal field split; after measurement retain the signed cross term."""
    actual_coefficients = len(j) * j @ basis.T
    oracle = coefficient_field(actual_coefficients, basis)
    predicted = coefficient_field(predicted_coefficients, basis)
    missing = oracle - j
    coefficient = predicted - oracle
    energy = float(np.sum(j**2))
    check = float(
        np.sum((predicted - j) ** 2) - np.sum(missing**2) - np.sum(coefficient**2)
    )
    if abs(check) > 1e-10 * max(energy, 1e-12):
        raise ValueError("Orthogonal field error split failed")
    actual_action = contractions(j, directions)
    d_action = contractions(missing, directions)
    c_action = contractions(coefficient, directions)
    ypred = contractions(predicted, directions)
    if np.max(abs(ypred - actual_action - d_action - c_action)) > 1e-10:
        raise ValueError("Action error split failed")
    cenergy = float(np.sum(actual_coefficients**2))
    return (
        {
            "field_basis_error": float(np.sum(missing**2) / energy),
            "field_coefficient_error": float(np.sum(coefficient**2) / energy),
            "field_split_error": check / energy,
            "coefficient_error_ratio": float(
                np.sum((predicted_coefficients - actual_coefficients) ** 2) / cenergy
            )
            if cenergy
            else float("nan"),
        },
        d_action,
        c_action,
        predicted,
    )


def verify_predictions(out):
    """Reject changed fitting/input/prediction assets before loading held targets."""
    m, f = verified(out), read(out / "prediction_freeze.json")
    if f["manifest_sha256"] != sha(out / "manifest.json") or f[
        "runtime_jobs_sha256"
    ] != sha(out / "private_jobs.json"):
        raise ValueError("Frozen manifest/jobs changed")
    for p, digest in f["files"].items():
        if sha(out / p) != digest:
            raise ValueError("Frozen prediction changed")
    return m


def analyze(out, name):
    """Read held gradients only after source-only predictions are fully frozen."""
    m = verify_predictions(out)
    prior_digest = read(SOURCE / "analysis/summary.json")["files"][
        "reader_comparison.csv"
    ]
    if sha(SOURCE / "analysis/reader_comparison.csv") != prior_digest:
        raise ValueError("Prior comparator table changed")
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "new_fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "prior_comparison_sha256": prior_digest,
        },
    )
    indices = [r["index"] for r in read(out / "private_membership.json")["authors"]]
    with np.load(out / "private_predictions.npz") as z:
        predictions = dict(z)
    actions, fields = [], []
    for q in m["queries"]:
        for i in indices:
            with np.load(SOURCE / f"private_inputs_{i}.npz") as z:
                h, g = z["hidden"], z["grain"]
            dirs = np.stack([direction(h, g, k)[0] for k in range(4)])
            with np.load(NATIVE / "private_packets" / f"a{i:02d}_{q}.npz") as z:
                j = z["body_gradient"].astype(float)
            for arm in ARMS:
                with np.load(out / q / f"private_model_{i}_{arm}.npz") as z:
                    basis = z["basis"]
                pieces, db, dc, pred = error_parts(
                    j, predictions[f"{q}__a{i}__{arm}"], basis, dirs
                )
                result = errors(j, pred, dirs)
                meta = {"query": q, "index": i, "arm": arm, "tokens": len(j)}
                fields.append(
                    meta
                    | pieces
                    | {
                        k: result[k]
                        for k in (
                            "field_error_ratio",
                            "field_actual_energy",
                            "field_error_energy",
                        )
                    }
                )
                for k in range(4):
                    actions.append(
                        meta
                        | {
                            "factor": f"whole:F{k + 1:02d}",
                            "actual": result["actual"][k],
                            "predicted": result["predicted"][k],
                            "basis_error": db[k],
                            "coefficient_error": dc[k],
                        }
                    )
    f, a = pd.DataFrame(fields), pd.DataFrame(actions)
    if (
        len(f) != 24
        or len(a) != 96
        or a.duplicated(["query", "arm", "index", "factor"]).any()
    ):
        raise ValueError("Incomplete paired comparison")
    old_actions = pd.read_csv(SOURCE / "analysis/private_actions.csv")
    old_actual = (
        old_actions[old_actions.arm == "content"]
        .sort_values(["query", "index", "factor"])
        .actual.to_numpy()
    )
    for arm in ARMS:
        actual = (
            a[a.arm == arm].sort_values(["query", "index", "factor"]).actual.to_numpy()
        )
        if not np.allclose(actual, old_actual, rtol=1e-12, atol=1e-12):
            raise ValueError("Old/new action targets differ")
    f.to_csv(dest / "private_fields.csv", index=False)
    a.to_csv(dest / "private_actions.csv", index=False)
    source = pd.read_csv(out / "private_source_fit.csv")
    rows, factors, accounts = [], [], []
    for (q, arm), g in a.groupby(["query", "arm"], sort=True):
        meta = {"query": q, "arm": arm, "authors": 4}
        fg = f[(f["query"] == q) & (f.arm == arm)]
        sg = source[(source["query"] == q) & (source.arm == arm)]
        centered = g[["actual", "predicted"]] - g.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        rows.append(
            meta
            | action_metrics(g.actual, g.predicted)
            | {
                "field_error_mean": float(fg.field_error_ratio.mean()),
                "coefficient_error_mean": float(fg.coefficient_error_ratio.mean()),
                "source_coefficient_error_mean": float(
                    sg.source_coefficient_error.mean()
                ),
                "factor_centered_action_r": action_metrics(
                    centered.actual, centered.predicted
                )["action_r"],
                "field_basis_error_mean": float(fg.field_basis_error.mean()),
                "field_coefficient_error_mean": float(
                    fg.field_coefficient_error.mean()
                ),
            }
        )
        for factor, h in g.groupby("factor", sort=True):
            factors.append(
                meta | {"factor": factor} | action_metrics(h.actual, h.predicted)
            )
        denom = float(np.sum(g.actual**2))
        db, dc = g.basis_error.to_numpy(), g.coefficient_error.to_numpy()
        total = float(np.sum((g.predicted - g.actual) ** 2) / denom)
        eb, ec, cross = (
            float(db @ db / denom),
            float(dc @ dc / denom),
            float(2 * db @ dc / denom),
        )
        if abs(total - eb - ec - cross) > 1e-10:
            raise ValueError("Measured error energy identity failed")
        accounts.append(
            meta
            | {
                "total_error": total,
                "basis_error_energy": eb,
                "coefficient_error_energy": ec,
                "signed_cross": cross,
                "identity_error": total - eb - ec - cross,
            }
        )
    pd.DataFrame(rows).to_csv(dest / "reader_comparison.csv", index=False)
    pd.DataFrame(factors).to_csv(dest / "factor_actions.csv", index=False)
    pd.DataFrame(accounts).to_csv(dest / "error_account.csv", index=False)
    src = (
        source.groupby(["query", "arm"], sort=True)
        .agg(
            fits=("held", "size"),
            coefficient_error_mean=("source_coefficient_error", "mean"),
            coefficient_error_min=("source_coefficient_error", "min"),
            coefficient_error_max=("source_coefficient_error", "max"),
            design_columns=("design_columns", "first"),
            extra_rank_min=("extra_rank", "min"),
            extra_rank_max=("extra_rank", "max"),
            extra_energy_mean=("extra_residual_energy_fraction", "mean"),
            pca_overlap_max=("pca_overlap_max", "max"),
            source_coordinate_abs_max=("source_coordinate_abs_max", "max"),
            held_coordinate_abs_max=("held_coordinate_abs_max", "max"),
            held_outside_fraction_mean=("held_coordinate_outside_fraction", "mean"),
            held_outside_fraction_max=("held_coordinate_outside_fraction", "max"),
        )
        .reset_index()
    )
    src.to_csv(dest / "source_fits.csv", index=False)
    # The old comparison belongs to the identical fields and query objectives.
    old = pd.read_csv(SOURCE / "analysis/reader_comparison.csv")
    old.to_csv(dest / "prior_comparators.csv", index=False)
    singular_rows = []
    for q in m["queries"]:
        for arm in ARMS:
            values = []
            for i in indices:
                with np.load(out / q / f"private_model_{i}_{arm}.npz") as z:
                    values.append(z["extra_singular"])
            for rank, values_at_rank in enumerate(np.stack(values).T, 1):
                singular_rows.append(
                    {
                        "query": q,
                        "arm": arm,
                        "rank": rank,
                        "mean": float(values_at_rank.mean()),
                        "min": float(values_at_rank.min()),
                        "max": float(values_at_rank.max()),
                        "units": "weighted_input_singular_value"
                        if arm == "residual_variance"
                        else "input_coefficient_cross_covariance_singular_value",
                    }
                )
    pd.DataFrame(singular_rows).to_csv(dest / "source_singulars.csv", index=False)
    names = [
        "source_singulars.csv",
        "reader_comparison.csv",
        "factor_actions.csv",
        "error_account.csv",
        "source_fits.csv",
        "prior_comparators.csv",
        "private_fields.csv",
        "private_actions.csv",
    ]
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "authors": 4,
            "queries": 3,
            "arms": 2,
            "new_fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "field_split_error_max": float(f.field_split_error.abs().max()),
            "action_split_error_max": max(abs(r["identity_error"]) for r in accounts),
            "files": {p: sha(dest / p) for p in names},
        },
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_query_residual_input_v1_r2")
    p.add_argument("--name", default="analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.name)

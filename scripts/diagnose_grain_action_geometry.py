#!/usr/bin/env python3
"""Post-result, no-fit separation of layout slopes, R paths and fold covariance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.level_response import correlation
from scripts.run_grain_action_geometry import PARENT, verify
from scripts.run_grain_joint_carrier_readout import check, donors_and_design
from scripts.run_grain_real_author_path import BASE, read, sha, write


def split_functional(origin, first_integral, second_integral):
    """Disjoint definition paths, not statistically independent matrix components."""
    a, b, c = (np.asarray(x, float) for x in (origin, first_integral, second_integral))
    if a.shape != b.shape or b.shape != c.shape or not np.isfinite([a, b, c]).all():
        raise ValueError("Finite aligned functional matrices required")
    return {"layout_origin": a, "retained_path": b - a, "body_condition": (c - b) / 2}


def grouped_covariance(x, y, groups):
    """Exact total covariance split, without using centered values as new predictions."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or g.shape != x.shape
        or not np.isfinite([x, y]).all()
    ):
        raise ValueError("Finite matched scalar values and groups required")
    mx, my = np.zeros_like(x), np.zeros_like(y)
    for k in np.unique(g):
        mask = g == k
        mx[mask], my[mask] = x[mask].mean(), y[mask].mean()
    within = float(np.mean((x - mx) * (y - my)))
    between = float(np.mean((mx - x.mean()) * (my - y.mean())))
    total = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = x.std() * y.std()
    return {
        "total_r": correlation(x, y),
        "within_fold_r": correlation(x - mx, y - my),
        "within_fold_covariance": within,
        "between_fold_covariance": between,
        "within_contribution_to_total_r": within / denominator
        if denominator > 1e-15
        else float("nan"),
        "between_contribution_to_total_r": between / denominator
        if denominator > 1e-15
        else float("nan"),
        "closure_error": abs(total - within - between),
    }


def run(output_name):
    """Read only frozen nodes and numeric actions, never source questionnaires."""
    out = BASE / "logic_action_geometry_v1"
    verify(out)
    for name, digest in read(out / "prediction_freeze.json")["files"].items():
        check(out / name, digest)
    dest = out / output_name
    dest.mkdir(exist_ok=False)
    _, design = donors_and_design()
    ref_rows = [i for i, r in enumerate(design) if r["role"] == "reference"]
    with np.load(out / "private_displacements.npz") as z:
        d = z["displacements"][ref_rows]
    components = []
    for i in ref_rows:
        r = design[i]
        paths = [
            f"private_nodes/ref_{r['index']:03d}_m0_t000.npz",
            f"private_nodes/ref_{r['index']:03d}_m0_integral.npz",
            f"private_nodes/ref_{r['index']:03d}_m1_integral.npz",
        ]
        arrays = []
        for name, key in zip(paths, ("gradient", "matrix", "matrix"), strict=True):
            with np.load(PARENT / name) as z:
                arrays.append(z[key].ravel().astype(float))
        components.append(split_functional(*arrays))
    parts = {k: np.array([row[k] for row in components]) for k in components[0]}
    total = sum(parts.values())
    total_deviation = total - total.mean(0)
    total_action_deviation = np.sum(total_deviation * d, axis=1)
    matrix_energy = float(np.mean(np.sum(total_deviation**2, axis=1)))
    action_energy = float(np.mean(total_action_deviation**2))
    rows, actions = [], []
    for name, part in parts.items():
        dev = part - part.mean(0)
        effect = np.sum(dev * d, axis=1)
        actions.append(effect)
        rows.append(
            {
                "component": name,
                "reference_authors": 4,
                "matrix_deviation_rms_norm": float(
                    np.sqrt(np.mean(np.sum(dev**2, axis=1)))
                ),
                "matrix_deviation_signed_share": float(
                    np.mean(np.sum(dev * total_deviation, axis=1)) / matrix_energy
                ),
                "own_action_deviation_rms": float(np.sqrt(np.mean(effect**2))),
                "action_deviation_signed_share": float(
                    np.mean(effect * total_action_deviation) / action_energy
                ),
                "own_component_action_mean": float(np.sum(part * d, axis=1).mean()),
                "own_component_action_rms": float(
                    np.sqrt(np.mean(np.sum(part * d, axis=1) ** 2))
                ),
            }
        )
    actions = np.array(actions)
    closure = float(np.max(abs(actions.sum(0) - total_action_deviation)))
    if closure > 1e-12:
        raise ValueError("Functional mechanism contraction did not close")
    pd.DataFrame(rows).to_csv(dest / "retained_condition_components.csv", index=False)
    data = pd.read_csv(out / "private_joint_geometry.csv", dtype={"author_id": str})
    rows = []
    for (pair, arm), g in data.groupby(["pair", "arm"]):
        rows.append(
            {
                "pair": pair,
                "arm": arm,
                **grouped_covariance(g.actual_amplitude, g.predicted_amplitude, g.fold),
            }
        )
    pd.DataFrame(rows).to_csv(dest / "joint_fold_covariance.csv", index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "new_fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "reference_action_closure_max": closure,
            "code_sha256": sha(Path(__file__)),
            "interpretation": "Origin gradient uses fixed M0/R0/PQ; only target layout varies. R-path and body-condition terms follow a declared ordered definition.",
        },
    )
    print(
        pd.read_csv(dest / "retained_condition_components.csv").to_string(index=False)
    )
    print(
        pd.DataFrame(rows)[
            [
                "pair",
                "arm",
                "total_r",
                "within_fold_r",
                "within_contribution_to_total_r",
                "between_contribution_to_total_r",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="condition_diagnostics")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Use a simple output name")
    run(args.name)

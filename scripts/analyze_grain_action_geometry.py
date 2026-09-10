#!/usr/bin/env python3
"""Model-action geometry first; old source criteria join only after numerical freeze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.level_response import correlation
from scripts.analyze_grain_joint_carrier_readout import (
    carrier_metrics,
    indices,
    interval,
)
from scripts.diagnose_grain_joint_carrier_readout import improvement_account
from scripts.run_grain_action_geometry import verify
from scripts.run_grain_joint_carrier_readout import check
from scripts.run_grain_real_author_path import BASE, read, sha, write


def joint_summary(data):
    """Evaluate within pair type, preserving same-author errors and output remainders."""
    rows = []
    for (pair, arm), g in data.groupby(["pair", "arm"]):
        g = g.sort_values("author_id")
        energy = float(g.actual_energy.sum())
        total_error = float(g.total_error.sum())
        a, b = g.actual_amplitude.to_numpy(), g.predicted_amplitude.to_numpy()
        ci = interval([correlation(a[ix], b[ix]) for ix in indices(g.author_id)])
        actual, pred = g.actual_internal.to_numpy(), g.predicted_internal.to_numpy()
        al, pl = g.actual_linear_output.to_numpy(), g.predicted_linear_output.to_numpy()
        denominator = float(np.sum(actual**2))
        rows.append(
            {
                "pair": pair,
                "arm": arm,
                "n": len(g),
                "actual_parallel_energy_fraction": float(
                    g.parallel_energy.sum() / energy
                ),
                "parallel_error_over_total_actual_energy": float(
                    g.parallel_error.sum() / energy
                ),
                "perpendicular_error_over_total_actual_energy": float(
                    g.perpendicular_error.sum() / energy
                ),
                "mse_over_zero": total_error / energy,
                "amplitude_r": correlation(a, b),
                "amplitude_r_low": ci[0],
                "amplitude_r_high": ci[1],
                "perpendicular_norm_r": correlation(
                    g.actual_perpendicular_norm, g.predicted_perpendicular_norm
                ),
                "actual_amplitude_mean": float(a.mean()),
                "predicted_amplitude_mean": float(b.mean()),
                "actual_amplitude_sd": float(a.std()),
                "predicted_amplitude_sd": float(b.std()),
                "exact_internal_output_mse_over_zero": float(
                    np.sum((actual - pred) ** 2) / denominator
                ),
                "linear_output_mse_over_actual_energy": float(
                    np.sum((al - pl) ** 2) / denominator
                ),
                "actual_taylor_error_rms": float(np.sqrt(np.mean((actual - al) ** 2))),
                "actual_internal_rms": float(np.sqrt(np.mean(actual**2))),
                "predicted_taylor_error_rms": float(np.sqrt(np.mean((pred - pl) ** 2))),
                "energy_closure_max": float(g.closure_error.max()),
            }
        )
    return pd.DataFrame(rows)


def carrier_comparisons(data):
    """Compare fixed calibration with both the prior and mean, on the same20 people."""
    rows = []
    chosen = data[
        (data.role == "evaluation") & (data.arm == "support_calibrated")
    ].sort_values("author_id")
    a, p = chosen.actual_retained.to_numpy(), chosen.predicted_retained.to_numpy()
    for arm in ("frozen_full", "reference_mean", "zero"):
        ctrl = (
            data[(data.role == "evaluation") & (data.arm == arm)]
            .set_index("author_id")
            .loc[chosen.author_id]
        )
        m = ctrl.predicted_retained.to_numpy()
        draws = indices(chosen.author_id)
        ci = interval(
            [
                improvement_account(a[ix], p[ix], m[ix])["gain_over_control"]
                for ix in draws
            ]
        )
        rows.append(
            {
                "control": arm,
                "n": len(a),
                **improvement_account(a, p, m),
                "gain_low": ci[0],
                "gain_high": ci[1],
            }
        )
    return pd.DataFrame(rows)


def source_links(carrier, joint, y):
    """Source labels describe frozen components; no model or sign is chosen here."""
    lookup = {(r.author_id, r.query): r.y_true for r in y.itertuples()}
    rows = []
    for arm, g in carrier[carrier.role == "evaluation"].groupby("arm"):
        g = g.sort_values("author_id")
        target = np.array([lookup[r.author_id, r.query] for r in g.itertuples()])
        for c in (
            "actual_retained",
            "predicted_retained",
            "actual_complete_b",
            "oracle_baseline_body_reconstruction",
        ):
            x = g[c].to_numpy()
            ci = interval(
                [correlation(x[ix], target[ix]) for ix in indices(g.author_id)]
            )
            rows.append(
                {
                    "query": "TF_cont",
                    "scope": "evaluation",
                    "arm": arm,
                    "component": c,
                    "n": len(g),
                    "r_y": correlation(x, target),
                    "low": ci[0],
                    "high": ci[1],
                }
            )
    for (pair, arm), g in joint.groupby(["pair", "arm"]):
        g = g.sort_values("author_id")
        target = np.array([lookup[r.author_id, r.query] for r in g.itertuples()])
        for c in (
            "actual_amplitude",
            "predicted_amplitude",
            "actual_perpendicular_norm",
            "predicted_perpendicular_norm",
            "actual_linear_output",
            "predicted_linear_output",
        ):
            x = g[c].to_numpy()
            ci = interval(
                [correlation(x[ix], target[ix]) for ix in indices(g.author_id)]
            )
            rows.append(
                {
                    "query": "neuroticism",
                    "scope": pair,
                    "arm": arm,
                    "component": c,
                    "n": len(g),
                    "r_y": correlation(x, target),
                    "low": ci[0],
                    "high": ci[1],
                }
            )
    return pd.DataFrame(rows)


def analyze(out, name):
    """Analyze complete frozen predictions with author-level bootstrap and no refitting."""
    manifest = verify(out)
    frozen = read(out / "prediction_freeze.json")
    if frozen["human_outcomes_read"] is not False:
        raise ValueError("Unsupervised criterion-link freeze required")
    for path, digest in frozen["files"].items():
        check(out / path, digest)
    dest = out / name
    dest.mkdir(exist_ok=False)
    carrier = pd.read_csv(
        out / "private_carrier_predictions.csv", dtype={"author_id": str}
    )
    joint = pd.read_csv(out / "private_joint_geometry.csv", dtype={"author_id": str})
    cm, jm, comp = (
        carrier_metrics(carrier),
        joint_summary(joint),
        carrier_comparisons(carrier),
    )
    cm.to_csv(dest / "carrier_metrics.csv", index=False)
    jm.to_csv(dest / "joint_geometry_metrics.csv", index=False)
    comp.to_csv(dest / "calibration_comparisons.csv", index=False)
    directions = read(out / "private_joint_directions.json")
    direction_stats = []
    for pair in ("adjacent", "distant"):
        v = np.array([r["direction"] for r in directions if r["pair"] == pair])
        cosines = (v @ v.T)[np.triu_indices(4, 1)]
        direction_stats.append(
            {
                "pair": pair,
                "cross_fold_direction_cosine_mean": float(cosines.mean()),
                "cross_fold_direction_cosine_min": float(cosines.min()),
            }
        )
    pd.DataFrame(direction_stats).to_csv(dest / "direction_agreement.csv", index=False)
    path = BASE / "source_anchor_diagnosis_v1/private_anchor_source_connection.csv"
    check(path, manifest["source_outcome_sha256"])
    y = pd.read_csv(
        path, usecols=["author_id", "query", "y_true"], dtype={"author_id": str}
    )
    if y.duplicated(["author_id", "query"]).any():
        raise ValueError("Unique source author/query link required")
    source_links(carrier, joint, y).to_csv(
        dest / "source_criterion_links.csv", index=False
    )
    summary = {
        "status": "COMPLETE",
        "cost": read(out / "run_status.json"),
        "support": read(out / "support_summary.json"),
        "dependency_accounts": read(out / "dependency_summary.json"),
        "source_labels_for_fitting": False,
        "numerical_hashes": {
            p.name: sha(p)
            for p in sorted(dest.glob("*.csv"))
            if not p.name.startswith("._")
        },
    }
    write(dest / "summary.json", summary)
    print(
        cm[cm.scope == "evaluation"][["arm", "error_energy_over_zero", "r"]].to_string(
            index=False
        )
    )
    print(
        comp[["control", "gain_over_control", "gain_low", "gain_high"]].to_string(
            index=False
        )
    )
    print(
        jm[jm.arm == "fourier_position"][
            [
                "pair",
                "actual_parallel_energy_fraction",
                "parallel_error_over_total_actual_energy",
                "perpendicular_error_over_total_actual_energy",
                "amplitude_r",
                "actual_taylor_error_rms",
            ]
        ].to_string(index=False)
    )
    print({"status": "ANALYSIS_COMPLETE", "new_neural_calls": 0, "fits": 1})


def main():
    """Keep the linked author values private and each analysis separately replayable."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=BASE / "logic_action_geometry_v1")
    p.add_argument("--name", default="analysis_r1")
    args = p.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Ignored path and simple analysis name required")
    analyze(out, args.name)


if __name__ == "__main__":
    main()

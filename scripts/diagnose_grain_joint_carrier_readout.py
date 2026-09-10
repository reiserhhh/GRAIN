#!/usr/bin/env python3
"""No-fit NEXT19/20 diagnostics: shared action versus person-specific modulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_joint_carrier_readout import indices, interval
from scripts.run_grain_joint_carrier_readout import check, donors_and_design, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def improvement_account(actual, predicted, control):
    """MSE gain over a frozen control equals alignment benefit minus added energy."""
    y, p, m = (np.asarray(x, float) for x in (actual, predicted, control))
    if y.shape != p.shape or y.shape != m.shape or not np.isfinite([y, p, m]).all():
        raise ValueError("Finite matched arrays required")
    residual, delta = y - m, p - m
    base = float(np.sum(residual**2))
    benefit, cost = float(2 * np.sum(residual * delta)), float(np.sum(delta**2))
    gain = float(np.sum((y - m) ** 2) - np.sum((y - p) ** 2))
    if abs(gain - benefit + cost) > 1e-10 * max(1.0, base):
        raise ValueError("Exact MSE accounting failed")
    return {
        "error_over_control": float(np.sum((y - p) ** 2) / base),
        "alignment_benefit_over_control": benefit / base,
        "added_energy_over_control": cost / base,
        "gain_over_control": gain / base,
        "closure_error": abs(gain - benefit + cost),
    }


def run(out):
    """Inspect frozen readouts and reference matrices without refitting or reading Y."""
    verify(out)
    for filename in ("joint_freeze.json", "carrier_freeze.json"):
        for name, digest in read(out / filename)["files"].items():
            check(out / name, digest)
    destination = out / "diagnostics"
    destination.mkdir(exist_ok=False)
    joint = read(out / "private_joint_predictions.json")
    control = {
        (r["author_id"], r["pair"]): r["predicted_mixed"]
        for r in joint
        if r["arm"] == "pair_train_mean"
    }
    rows = []
    for arm in ("fourier_bag", "fourier_position"):
        for pair in ("all", "adjacent", "distant"):
            g = sorted(
                [
                    r
                    for r in joint
                    if r["arm"] == arm and (pair == "all" or pair == r["pair"])
                ],
                key=lambda r: (r["author_id"], r["pair"]),
            )
            y, p = [
                np.array([r[k] for r in g]) for k in ("actual_mixed", "predicted_mixed")
            ]
            m = np.array([control[r["author_id"], r["pair"]] for r in g])
            ci = interval(
                [
                    improvement_account(y[ix], p[ix], m[ix])["gain_over_control"]
                    for ix in indices([r["author_id"] for r in g])
                ]
            )
            row = {
                "query": "neuroticism",
                "scope": pair,
                "arm": arm,
                **improvement_account(y, p, m),
                "gain_low": ci[0],
                "gain_high": ci[1],
            }
            row["positive_folds"] = sum(
                improvement_account(y[ix], p[ix], m[ix])["gain_over_control"] > 0
                for fold in range(4)
                for ix in [np.array([r["fold"] == fold for r in g])]
            )
            rows.append(row)
    carrier = pd.read_csv(
        out / "private_carrier_predictions.csv", dtype={"author_id": str}
    )
    control = carrier[
        (carrier.role == "evaluation") & (carrier.arm == "reference_mean")
    ].set_index("author_id")
    for arm, g in carrier[carrier.role == "evaluation"].groupby("arm"):
        g = g.sort_values("author_id")
        y, p = g.actual_retained.to_numpy(), g.predicted_retained.to_numpy()
        m = control.loc[g.author_id, "predicted_retained"].to_numpy()
        ci = interval(
            [
                improvement_account(y[ix], p[ix], m[ix])["gain_over_control"]
                for ix in indices(g.author_id)
            ]
        )
        rows.append(
            {
                "query": "TF_cont",
                "scope": "evaluation",
                "arm": arm,
                **improvement_account(y, p, m),
                "gain_low": ci[0],
                "gain_high": ci[1],
                "actual_sd": float(y.std()),
                "predicted_sd": float(p.std()),
                "mean_bias": float(p.mean() - y.mean()),
            }
        )
    pd.DataFrame(rows).to_csv(
        destination / "control_relative_accounts.csv", index=False
    )
    frozen = read(out / "carrier_nodes_freeze.json")
    for name, digest in frozen["files"].items():
        check(out / name, digest)
    _, design = donors_and_design()
    matrices = []
    for row in [r for r in design if r["role"] == "reference"]:
        arms = []
        for c in (0, 1):
            with np.load(
                out / "private_nodes" / f"ref_{row['index']:03d}_m{c}_integral.npz"
            ) as z:
                arms.append(z["matrix"])
        matrices.append(arms)
    matrices = np.array(matrices)
    avg = matrices.mean((0, 1))
    user_average = matrices.mean(1)
    common = float(np.sum(avg**2))
    between = float(np.mean(np.sum((user_average - avg) ** 2, axis=(1, 2))))
    within = float(
        np.mean(np.sum((matrices - user_average[:, None]) ** 2, axis=(2, 3)))
    )
    total = float(np.mean(np.sum(matrices**2, axis=(2, 3))))
    closure = abs(total - common - between - within)
    if closure > 1e-12 * max(1.0, total):
        raise ValueError("Matrix energy account failed")
    summary = {
        "reference_matrices": 8,
        "reference_authors": 4,
        "common_energy_fraction": common / total,
        "between_reference_energy_fraction": between / total,
        "within_body_condition_energy_fraction": within / total,
        "energy_closure_error": closure,
        "interpretation": "Frobenius matrix energy under this reference design; not human variance or an independent layout intervention",
        "human_labels_read": False,
        "new_neural_calls": 0,
        "fits": 0,
        "code_sha256": sha(Path(__file__)),
        "control_account_sha256": sha(destination / "control_relative_accounts.csv"),
    }
    write(destination / "diagnostic_summary.json", summary)
    print(summary)
    print(
        pd.DataFrame(rows)[
            ["query", "scope", "arm", "gain_over_control", "gain_low", "gain_high"]
        ].to_string(index=False)
    )


def main():
    """Run the declared post-result diagnostic once, without altering the first analyses."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_joint_carrier_readout_v1"
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Ignored research directory required")
    run(out)


if __name__ == "__main__":
    main()

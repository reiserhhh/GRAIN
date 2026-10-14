#!/usr/bin/env python3
"""Independently recheck saved arithmetic and exact choice/design witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]


def check_metrics(predictions, metrics, keys):
    """Reaggregate saved parameter estimates without importing the experiment code."""
    error = 0.0
    grouped = predictions.groupby(keys, dropna=False)
    for row in metrics.to_dict("records"):
        group = grouped.get_group(tuple(row[k] for k in keys))
        assert len(group) == row["scheduled"]
        valid = group[np.isfinite(group.prediction)]
        assert len(valid) == row["n"]
        for name in ("truth", "reader_truth"):
            part = valid[np.isfinite(valid[name])]
            if len(part):
                p, t = part.prediction.to_numpy(), part[name].to_numpy()
                calculated = {
                    "rmse": np.sqrt(((p - t) ** 2).mean()),
                    "bias": (p - t).mean(),
                }
                if len(part) > 2 and min(np.std(p), np.std(t)) > 1e-12:
                    calculated["r"] = np.corrcoef(p, t)[0, 1]
                for metric, value in calculated.items():
                    error = max(error, abs(value - row[name + "_" + metric]))
    assert error < 1e-10
    return error


def check_pilots(output, fitted):
    """Reconstruct source-only moment estimates from saved observed evidence."""
    error = 0.0
    for key, fit in fitted.items():
        y = np.load(output / f"pilot_{key}.npy")
        means = y.mean(axis=2)
        within = np.mean(
            np.sum((y - means[..., None]) ** 2, axis=2) / (y.shape[2] - 1), axis=0
        )
        between = (
            (means - means.mean(axis=0)).T @ (means - means.mean(axis=0)) / (len(y) - 1)
        )
        values, vectors = np.linalg.eigh(between - np.diag(within / y.shape[2]))
        covariance = vectors @ np.diag(np.maximum(values, 1e-6)) @ vectors.T
        for expected, actual in (
            (within, fit["noise"]),
            (means.mean(axis=0), fit["mean"]),
            (covariance, fit["covariance"]),
        ):
            error = max(error, np.max(abs(expected - actual)))
    assert error < 1e-10
    return error


def check_saved_scores(output, frame, fits):
    """Check 200 deterministic saved scores by Gaussian block conditioning."""
    chosen = frame[frame.policy != "oracle_full"].sample(200, random_state=91395)
    targets = {
        "level": np.array([0.5, 0.5]),
        "response": np.array([-0.5, 0.5]),
        "positive_condition": np.array([0, 1]),
    }
    maximum = 0.0
    for row in chosen.itertuples():
        key = f"{row.query}_{row.seed}"
        pool = np.load(output / f"held_{key}_{row.world}.npz")["evidence"][row.author]
        counts = np.array([row.minus_count, row.plus_count], dtype=int)
        cell = np.r_[np.zeros(counts[0], int), np.ones(counts[1], int)]
        y = np.r_[pool[0, : counts[0]], pool[1, : counts[1]]]
        if row.policy == "repeat_naive":
            cell, y = np.tile(cell, 2), np.tile(y, 2)
        fit = fits[key]
        mean, tau, noise = map(
            np.asarray, (fit["mean"], fit["covariance"], fit["noise"])
        )
        x = np.eye(2)[cell]
        gain = tau @ x.T @ np.linalg.inv(x @ tau @ x.T + np.diag(noise[cell]))
        expected = mean + gain @ (y - x @ mean)
        covariance = tau - gain @ x @ tau
        target = targets[row.target]
        maximum = max(
            maximum,
            abs(target @ expected - row.prediction),
            abs(target @ covariance @ target - row.variance),
        )
    assert maximum < 1e-10
    return maximum


def exact_witnesses():
    """Enumerate a choice tree and a state-selection counterexample."""
    p = expit(np.array([0.7, -0.5, 1.2]))
    information = np.zeros((3, 3))
    for root in (0, 1):
        node = root + 1
        for tail in (0, 1):
            probability = (p[0] if root else 1 - p[0]) * (
                p[node] if tail else 1 - p[node]
            )
            score = np.zeros(3)
            score[0], score[node] = root - p[0], tail - p[node]
            information += probability * np.outer(score, score)
    expected = np.diag(
        [p[0] * (1 - p[0]), (1 - p[0]) * p[1] * (1 - p[1]), p[0] * p[2] * (1 - p[2])]
    )
    np.testing.assert_allclose(information, expected, atol=1e-15)
    # At zero root log odds, state +/-1 shifts choice log odds by +/-2.
    response_bias = 0.3 * np.tanh(1.0)
    return {
        "tree_fisher": information.tolist(),
        "tail_increment_given_root": float(information[1, 1] + information[2, 2]),
        "fixed_assignment_choice_information": 0,
        "opportunity_nonidentifiability": "(gamma+d, offset-d) gives identical choice probabilities",
        "state_selected_response_bias_at_zero_logit": float(response_bias),
        "assigned_condition_state_response_bias": 0,
        "scope": "Exact specified finite generative models, not new psychometric axioms",
    }


def decision_diagnostics(output):
    """Summarize source prediction versus evaluation-only allocation risk surfaces."""
    risk = pd.read_csv(output / "allocation_risk.csv")
    rows = []
    for group, frame in risk.groupby(["query", "world", "target", "seed"]):
        selected = frame[frame.selected].iloc[0]
        oracle = frame[frame.oracle_selected].iloc[0]
        rows.append(
            {
                **dict(zip(["query", "world", "target", "seed"], group)),
                "selected_minus_count": int(selected.minus_count),
                "oracle_minus_count": int(oracle.minus_count),
                "source_predicted_mse": selected.predicted_risk,
                "actual_reader_mse": selected.actual_reader_mse,
                "risk_surface_rank_rho": frame.predicted_risk.corr(
                    frame.actual_reader_mse, method="spearman"
                ),
                "hindsight_allocation_regret": selected.actual_reader_mse
                - frame.actual_reader_mse.min(),
                "oracle_design_difference_same_scorer": selected.actual_reader_mse
                - oracle.actual_reader_mse,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "decision_diagnostics.csv", index=False)
    return len(result)


def main():
    """Validate hashes, metrics, source learning and selected scores; save audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    summary = json.loads((output / "summary.json").read_text())
    contract = json.loads((output / "contract.json").read_text())
    for name, expected in summary["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, (
            name
        )
    for name, expected in contract["input_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    choice, design = [
        pd.read_csv(output / f"{name}_predictions.csv") for name in ("choice", "design")
    ]
    report = {"status": "pass", "hashed_outputs": len(summary["files"]), "checks": {}}
    for name, frame, keys in (
        ("choice", choice, ["query", "world", "design", "estimator", "target"]),
        ("design", design, ["query", "world", "policy", "target"]),
    ):
        report["checks"][name + "_metric_max_error"] = check_metrics(
            frame, pd.read_csv(output / f"{name}_metrics.csv"), keys
        )
        report["checks"][name + "_seed_metric_max_error"] = check_metrics(
            frame, pd.read_csv(output / f"{name}_metrics_by_seed.csv"), keys + ["seed"]
        )
    fits = json.loads((output / "source_fits.json").read_text())
    report["checks"]["source_moment_max_error"] = check_pilots(output, fits)
    report["checks"]["independent_block_conditioning_max_error"] = check_saved_scores(
        output, design, fits
    )
    assigned = choice[
        (choice.design == "fixed") & choice.target.str.startswith("choice_")
    ]
    assert assigned.prediction.isna().all()
    assert (design.reports == contract["config"]["design_budget"]).all()
    assert (
        design[design.policy == "repeat_aware"].unique_events * 2
        == design[design.policy == "repeat_aware"].reports
    ).all()
    assert (
        np.max(
            abs(design.prediction - design.observed_term - design.borrowed_mean_term)
        )
        < 1e-12
    )
    report["checks"]["missing_choice_not_imputed"] = True
    report["checks"]["same_report_budget"] = True
    report["checks"]["source_fit_inputs"] = (
        "observed source pilot only; no truth or held-author input"
    )
    witnesses = exact_witnesses()
    (output / "exact_witnesses.json").write_text(json.dumps(witnesses, indent=2) + "\n")
    report["exact_witnesses"] = witnesses
    report["decision_diagnostic_rows"] = decision_diagnostics(output)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S11: distill fixed N intervention components using existing E5 means only."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_functional_basis import author_folds
from scripts.analyze_grain_multiquery_sensitivity import corr
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION, CURVE, covariance
from scripts.run_grain_student_learning_curve import feature_cache
from scripts.run_grain_supervised_student import coordinates, fit_student, geometry

ALPHAS = (1.0, 10.0, 100.0)
TARGETS = ("new_teacher", "mean_score", "configuration_response")
METHODS = ("pca64", "pls2")


def specification(method):
    """Map fixed names to the already tested student-coordinate implementation."""
    if method == "pca64":
        return "pca64", 0
    if method == "pls2":
        return "pls", 2
    raise ValueError("Unknown frozen coordinate family")


def select(x, target, names, method, salt):
    """Choose alpha on target-only inner folds; dimensions are not searched."""
    family, rank = specification(method)
    folds = author_folds(names, 3, salt)
    losses = {a: [] for a in ALPHAS}
    for fold in range(3):
        train, valid = folds != fold, folds == fold
        fitted = geometry(x[train], target[train], family, max(rank, 1))
        z, zv = coordinates(x[train], fitted, rank), coordinates(x[valid], fitted, rank)
        for alpha in ALPHAS:
            fit = Ridge(alpha=alpha).fit(z, target[train])
            losses[alpha].append(float(np.mean((fit.predict(zv) - target[valid]) ** 2)))
    return min(losses, key=lambda a: (np.mean(losses[a]), a))


def common_operator_error(x, first, second, xt):
    """Check fixed PCA/Ridge linearity: learning two summands cannot create information."""
    predictions = []
    for y in (first, second, first + second):
        coef, intercept = fit_student(x, y, "pca64", 0, 10.0)
        predictions.append(xt @ coef + intercept)
    return float(np.max(np.abs(predictions[0] + predictions[1] - predictions[2])))


def main():
    """Fit finite N-only component students; freeze OOF rows before joining source Y."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private outputs must remain ignored")
    component_path = args.source / "private_components_before_outcomes.csv"
    source_summary = read(args.source / "summary.json")
    if (
        source_summary["status"] != "COMPLETE"
        or sha(component_path) != source_summary["components_sha256"]
    ):
        raise ValueError("Complete frozen S9 source required")
    targets = pd.read_csv(component_path, dtype={"author_id": str})
    targets = targets[targets["query"] == "neuroticism"].set_index("author_id")
    if len(targets) != 213 or not targets.index.is_unique:
        raise ValueError("Require the full fixed N cohort")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_models").mkdir()
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S11_N_CONTRAST_STUDENT_V1",
            "script_sha256": sha(__file__),
            "source_components_sha256": sha(component_path),
            "methods": METHODS,
            "targets": TARGETS,
            "alphas": ALPHAS,
            "authors": 213,
            "selection": "target-only inner3 MSE, outer5 s4_outer",
            "scope": "post-S9 exposed-source discovery; no outcome fitting",
            "max_seconds": 600,
            "new_forwards": 0,
        },
    )
    start = time.monotonic()
    ids, features = feature_cache(CONNECTION, args.panel)
    ix = np.array([i for i, uid in enumerate(ids) if uid in targets.index])
    names, x = ids[ix], features["neuroticism"]["mean"][ix]
    if len(ix) != 213:
        raise ValueError("Incomplete E5 source features")
    targets = targets.loc[names]
    folds = author_folds(names, 5, "s4_outer")
    old = pd.read_csv(
        CURVE / "private_oof_before_outcomes.csv", dtype={"author_id": str, "size": str}
    )
    old = old[
        (old["query"] == "neuroticism")
        & (old.family == "mean")
        & (old["size"] == "all")
    ].set_index("author_id")
    if not np.array_equal(folds, old.loc[names, "fold"].to_numpy()):
        raise ValueError("S4 outer folds changed")
    rows, selections, linear_errors = [], [], []
    with threadpool_limits(limits=2):
        for fold in range(5):
            train, test = folds != fold, folds == fold
            predicted = {}
            for method in METHODS:
                family, rank = specification(method)
                for target in TARGETS:
                    y = targets[target].to_numpy()
                    alpha = select(
                        x[train], y[train], names[train], method, f"s11_inner_{fold}"
                    )
                    coef, intercept = fit_student(
                        x[train], y[train], family, rank, alpha
                    )
                    prediction = x[test] @ coef + intercept
                    model_path = (
                        out / "private_models" / f"{method}_{target}_fold{fold}.npz"
                    )
                    np.savez_compressed(model_path, coef=coef, intercept=intercept)
                    with np.load(model_path, allow_pickle=False) as saved:
                        replay = x[test] @ saved["coef"] + saved["intercept"]
                    if not np.allclose(prediction, replay, atol=1e-12):
                        raise ValueError("Student replay failed")
                    predicted[method, target] = prediction
                    selections.append(
                        {
                            "fold": fold,
                            "method": method,
                            "target": target,
                            "alpha": alpha,
                        }
                    )
                    rows.extend(
                        {
                            "author_id": uid,
                            "fold": fold,
                            "method": method,
                            "target": target,
                            "teacher": float(t),
                            "prediction": float(p),
                        }
                        for uid, t, p in zip(
                            names[test], y[test], prediction, strict=True
                        )
                    )
                summed = (
                    predicted[method, "mean_score"]
                    + predicted[method, "configuration_response"]
                )
                rows.extend(
                    {
                        "author_id": uid,
                        "fold": fold,
                        "method": method,
                        "target": "component_sum",
                        "teacher": float(t),
                        "prediction": float(p),
                    }
                    for uid, t, p in zip(
                        names[test],
                        targets.new_teacher.to_numpy()[test],
                        summed,
                        strict=True,
                    )
                )
            linear_errors.append(
                common_operator_error(
                    x[train],
                    targets.mean_score.to_numpy()[train],
                    targets.configuration_response.to_numpy()[train],
                    x[test],
                )
            )
            if time.monotonic() - start > 600:
                raise TimeoutError("Ten-minute student fit cap reached")
    if max(linear_errors) > 1e-10:
        raise ValueError("Fixed linear learning operator failed additive control")
    frame = pd.DataFrame(rows)
    if (
        len(frame) != 213 * 8
        or frame.duplicated(["author_id", "method", "target"]).any()
    ):
        raise ValueError("Incomplete or duplicate student OOF rows")
    oof = out / "private_oof_before_outcomes.csv"
    frame.to_csv(oof, index=False)
    write(out / "oof_frozen.json", {"sha256": sha(oof), "outcomes_used": False})
    pd.DataFrame(selections).to_csv(out / "selected_models.csv", index=False)
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(labels, on="author_id", validate="many_to_one")
    frame = frame.merge(
        targets[["student"]],
        left_on="author_id",
        right_index=True,
        validate="many_to_one",
    )
    metrics, comparisons = [], []
    for (method, target), g in frame.groupby(["method", "target"]):
        external = scalar_metrics(g.y_true, g.prediction)
        metrics.append(
            {
                "method": method,
                "target": target,
                "authors": len(g),
                "teacher_r": corr(g.teacher, g.prediction),
                "teacher_mse": float(np.mean((g.teacher - g.prediction) ** 2)),
                "external_r": external["pearson_r"],
                "external_p": external["pearson_p"],
                "retained_label_covariance": covariance(g.y_true, g.prediction),
            }
        )
        comparisons.append(
            {
                "method": method,
                "target": target,
                "control": "frozen_s4_student",
                **paired_intervals(g.y_true, g.prediction, g.student, False),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_intervals.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha(out / "manifest.json"),
            "oof_sha256": sha(oof),
            "wall_seconds": time.monotonic() - start,
            "new_forwards": 0,
            "questionnaire_fit": False,
            "fixed_operator_max_additivity_error": max(linear_errors),
            "metrics": metrics,
            "paired_intervals": comparisons,
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(
        {
            "seconds": time.monotonic() - start,
            "max_additivity_error": max(linear_errors),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

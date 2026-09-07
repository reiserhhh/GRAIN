#!/usr/bin/env python3
"""S0: nested author-CV comparison of variance and supervised response coordinates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_real_author_path import load_packet, prediction_quality
from scripts.run_grain_real_author_path import key, sha, write

BASE = ROOT / "results/grain_vector_mechanisms"
METHODS = ("svd_ridge", "full_ridge", "pls_ridge")
RANKS, RIDGES = (2, 4, 8), (1.0, 10.0, 100.0)


def author_folds(groups, count=5, salt="s0_outer"):
    """Keep every view, edit and dose of an author in one deterministic fold."""
    names = sorted(set(groups), key=lambda value: key(value, salt))
    if len(names) < count:
        raise ValueError("Too few authors for grouped CV")
    mapping = {name: j % count for j, name in enumerate(names)}
    return np.asarray([mapping[name] for name in groups])


def response_basis(x, y, method, rank=8):
    """Fit uncentered SVD or PLS1 directions, without using validation outcomes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("Invalid response design")
    if not np.isfinite(y).all() or rank < 1:
        raise ValueError("Finite response and positive rank required")
    if method == "full_ridge":
        return None
    if method == "svd_ridge":
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        k = min(rank, int(np.sum(s > max(s[0], 1.0) * 1e-10)))
        return vt[:k].T
    if method != "pls_ridge":
        raise ValueError("Unknown functional basis")
    residual, target = x.copy(), y.copy()
    weights, loadings = [], []
    for _ in range(min(rank, *x.shape)):
        w = residual.T @ target
        size = np.linalg.norm(w)
        if size <= 1e-12:
            break
        w /= size
        t = residual @ w
        energy = float(t @ t)
        if energy <= 1e-12:
            break
        p, c = residual.T @ t / energy, target @ t / energy
        weights.append(w)
        loadings.append(p)
        residual -= np.outer(t, p)
        target -= t * c
    if not weights:
        return np.empty((x.shape[1], 0))
    w, p = np.column_stack(weights), np.column_stack(loadings)
    rotations = w @ np.linalg.pinv(p.T @ w)
    return np.linalg.qr(rotations, mode="reduced")[0]


def fit_estimator(x, y, basis, ridge):
    """Fit an origin-preserving RMS-scaled Ridge in selected training coordinates."""
    if ridge <= 0:
        raise ValueError("Positive ridge required")
    z = x if basis is None else x @ basis
    scale = np.maximum(np.sqrt(np.mean(z**2, axis=0)), 1e-10)
    z = z / scale
    if z.shape[1] <= len(z):
        coef = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ y)
    else:
        coef = z.T @ np.linalg.solve(z @ z.T + ridge * np.eye(len(z)), y)
    return {"basis": basis, "scale": scale, "coef": coef}


def predict(model, x):
    """Apply frozen coordinates/scales without recentering evaluation authors."""
    z = x if model["basis"] is None else x @ model["basis"]
    return (z / model["scale"]) @ model["coef"]


def candidate_predictions(x, y, groups, method, salt):
    """Produce inner-CV predictions for each finite rank/Ridge candidate."""
    folds = author_folds(groups, 3, salt)
    ranks = (0,) if method == "full_ridge" else RANKS
    predictions = {
        (rank, ridge): np.zeros(len(y)) for rank in ranks for ridge in RIDGES
    }
    for fold in range(3):
        train = folds != fold
        basis = response_basis(x[train], y[train], method, max(RANKS))
        for rank in ranks:
            selected = None if basis is None else basis[:, :rank]
            for ridge in RIDGES:
                model = fit_estimator(x[train], y[train], selected, ridge)
                predictions[rank, ridge][~train] = predict(model, x[~train])
    return predictions


def nested_predictions(x, y, groups, task):
    """Choose hyperparameters only within outer-training authors and return OOF rows."""
    outer = author_folds(groups)
    output = {method: np.zeros(len(y)) for method in METHODS}
    selections = []
    for fold in range(5):
        train = outer != fold
        for method in METHODS:
            inner = candidate_predictions(
                x[train], y[train], groups[train], method, f"s0_inner_{fold}"
            )
            rank, ridge = min(
                inner, key=lambda spec: (np.mean((inner[spec] - y[train]) ** 2), spec)
            )
            basis = response_basis(x[train], y[train], method, max(rank, 1))
            model = fit_estimator(x[train], y[train], basis, ridge)
            output[method][~train] = predict(model, x[~train])
            selections.append(
                {
                    "task": task,
                    "fold": fold,
                    "method": method,
                    "rank": rank,
                    "ridge": ridge,
                    "inner_mse": float(np.mean((inner[rank, ridge] - y[train]) ** 2)),
                    "training_authors": len(set(groups[train])),
                    "test_authors": len(set(groups[~train])),
                }
            )
        print(f"S0 {task} outer fold {fold + 1}/5 complete", flush=True)
    return output, selections, outer


def paired_interval(y, candidate, baseline, groups, draws=2000):
    """Bootstrap paired author-average MSE differences, not duplicated dose rows."""
    frame = pd.DataFrame(
        {"author": groups, "difference": (y - candidate) ** 2 - (y - baseline) ** 2}
    )
    values = frame.groupby("author").difference.mean().to_numpy()
    rng = np.random.default_rng(20260908)
    sampled = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return {
        "mse_delta_vs_svd": float(values.mean()),
        "mse_delta_lower": float(np.quantile(sampled, 0.025)),
        "mse_delta_upper": float(np.quantile(sampled, 0.975)),
    }


def main():
    """Analyze existing intervention caches only and save all private OOF predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--packets",
        type=Path,
        nargs="+",
        default=[BASE / "d2_real_author_path_r1", BASE / "d3b_fresh_author_path_r1"],
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private outputs must stay in ignored results/")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    frames, vectors, targets, seen = [], [], [], set()
    for folder in args.packets:
        _, rows, x, y = load_packet(folder)
        ids = set(rows.author_id)
        if seen & ids:
            raise ValueError("Source packets overlap in authors")
        seen |= ids
        frames.append(rows.assign(source_packet=folder.name))
        vectors.append(x)
        targets.append(y)
    rows, x, y = (
        pd.concat(frames, ignore_index=True),
        np.vstack(vectors),
        np.concatenate(targets),
    )
    write(
        out / "manifest.json",
        {
            "status": "S0_NESTED_AUTHOR_CV",
            "authors": len(seen),
            "rows": len(rows),
            "all_existing_authors_are_development": True,
            "neural_forwards": 0,
            "psychological_labels_read": False,
            "script_sha256": sha(__file__),
            "packets": {str(p): sha(p / "manifest.json") for p in args.packets},
            "models": METHODS,
            "ranks": RANKS,
            "ridges": RIDGES,
            "selection": "minimum inner author-CV MSE; origin preserving; 5 outer/3 inner",
            "scaling": "training RMS in each estimator's coordinates",
        },
    )
    metrics, selections, private = [], [], []
    with threadpool_limits(limits=2):
        for task in sorted(rows.task.unique()):
            ix = np.flatnonzero((rows.task == task).to_numpy())
            frame = rows.iloc[ix]
            groups = frame.author_id.to_numpy()
            predictions, chosen, folds = nested_predictions(x[ix], y[ix], groups, task)
            selections.extend(chosen)
            for method, p in predictions.items():
                metrics.append(
                    {
                        "task": task,
                        "method": method,
                        "authors": len(set(groups)),
                        "rows": len(ix),
                        **prediction_quality(y[ix], p),
                        **paired_interval(y[ix], p, predictions["svd_ridge"], groups),
                    }
                )
                part = frame[
                    ["author_id", "view", "task", "donor", "alpha", "source_packet"]
                ].copy()
                part["fold"], part["method"] = folds, method
                part["response"], part["prediction"] = y[ix], p
                private.append(part)
    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selected_models.csv", index=False)
    pd.concat(private, ignore_index=True).to_csv(
        out / "author_cv_predictions.csv", index=False
    )
    summary = {
        "status": "COMPLETE",
        "seconds": time.monotonic() - start,
        "metrics": metrics,
        "neural_forwards": 0,
        "labels_read": False,
    }
    write(out / "summary.json", summary)
    report = "# S0 functional basis comparison\n\n"
    report += "Existing96 authors are development; nested author CV, no questionnaire targets.\n\n"
    report += table.to_markdown(index=False, floatfmt=".5f")
    report += "\n\nNegative MSE delta favors the candidate over SVD. Response r is not trait r.\n"
    report += "RMS scaling differs by coordinate system; compare whole estimators.\n"
    report += "Bootstrap intervals describe paired OOF errors, conditional on these fitted folds.\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

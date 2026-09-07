#!/usr/bin/env python3
"""S5A: teacher-supervised versus variance coordinates on fixed E5 author vectors."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_functional_basis import author_folds, response_basis
from scripts.analyze_grain_teacher_student_bridge import (
    covariance_split,
    paired_intervals,
)
from scripts.run_grain_real_author_path import BASE, key, sha, write
from scripts.run_grain_student_learning_curve import feature_cache

METHODS = ("pca64", "full", "pls")
RANKS = (2, 4, 8, 16, 32)
ALPHAS = (1.0, 10.0, 100.0)


def geometry(x, teacher, method, rank=32):
    """Fit training-only standardization and optional orthonormal projection."""
    x, teacher = np.asarray(x, float), np.asarray(teacher, float)
    if x.ndim != 2 or teacher.shape != (len(x),):
        raise ValueError("Paired training matrix and teacher required")
    if not np.isfinite(x).all() or not np.isfinite(teacher).all():
        raise ValueError("Finite training inputs required")
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = (x - center) / scale
    if method == "pca64":
        pca = PCA(
            n_components=min(64, len(x) - 1, x.shape[1]),
            svd_solver="randomized",
            random_state=731,
        ).fit(z)
        basis = pca.components_.T
    elif method == "pls":
        basis = response_basis(z, teacher - teacher.mean(), "pls_ridge", rank)
    elif method == "full":
        basis = None
    else:
        raise ValueError("Unknown student geometry")
    return center, scale, basis


def coordinates(x, fitted, rank=0):
    """Apply fixed training coordinates without adapting to the test batch."""
    center, scale, basis = fitted
    z = (x - center) / scale
    if basis is None:
        return z
    return z @ (basis[:, :rank] if rank else basis)


def inner_select(x, teacher, names, method, salt):
    """Choose rank/alpha solely by mean inner-fold teacher squared error."""
    folds = author_folds(names, 3, salt)
    ranks = RANKS if method == "pls" else (0,)
    errors = {(r, a): [] for r in ranks for a in ALPHAS}
    for fold in range(3):
        train, valid = folds != fold, folds == fold
        fitted = geometry(x[train], teacher[train], method, max(RANKS))
        for rank in ranks:
            z, zv = (
                coordinates(x[train], fitted, rank),
                coordinates(x[valid], fitted, rank),
            )
            for alpha in ALPHAS:
                head = Ridge(alpha=alpha).fit(z, teacher[train])
                errors[rank, alpha].append(
                    float(np.mean((head.predict(zv) - teacher[valid]) ** 2))
                )
    best = min(errors, key=lambda spec: (np.mean(errors[spec]), spec))
    return *best, float(np.mean(errors[best]))


def fit_student(x, teacher, method, rank, alpha):
    """Return an equivalent affine map in original coordinates for portable scoring."""
    fitted = geometry(x, teacher, method, max(rank, 1))
    z = coordinates(x, fitted, rank)
    head = Ridge(alpha=alpha).fit(z, teacher)
    center, scale, basis = fitted
    b = (
        head.coef_
        if basis is None
        else (basis[:, :rank] if rank else basis) @ head.coef_
    )
    coef = b / scale
    intercept = float(head.intercept_ - center @ coef)
    return coef, intercept


def teacher_arrays(ids, matrix, targets, query):
    """Join already authorized author membership to teacher targets, not outcomes."""
    native = targets[targets["query"] == query].set_index("author_id")
    if not native.index.is_unique or len(set(ids)) != len(ids):
        raise ValueError("Duplicate source authors")
    ix = np.array([j for j, uid in enumerate(ids) if uid in native.index])
    if len(ix) != len(native):
        raise ValueError("Missing teacher encoder cache")
    names = ids[ix]
    return names, matrix[ix], native.loc[names, "native_query"].to_numpy()


def freeze_and_report(out, rows, selections, source, start):
    """Seal all OOF estimates before loading outcomes; report every fixed arm."""
    frame = pd.DataFrame(rows)
    if frame.duplicated(["query", "method", "author_id"]).any():
        raise ValueError("Duplicated OOF row")
    path = out / "private_oof_before_outcomes.csv"
    frame.to_csv(path, index=False)
    write(out / "oof_frozen.json", {"sha256": sha(path), "outcome_selection": False})
    pd.DataFrame(selections).to_csv(out / "selected_models.csv", index=False)
    outcomes = pd.read_csv(
        source,
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true", "e5_large"],
    )
    frame = frame.merge(outcomes, on=["author_id", "query"], validate="many_to_one")
    if not np.isfinite(frame[["teacher", "prediction", "y_true"]]).all().all():
        raise ValueError("Nonfinite paired outcomes")
    metrics, moments, intervals = [], [], []
    for (query, method), g in frame.groupby(["query", "method"]):
        binary = query == "TF_cont"
        external = scalar_metrics(g.y_true, g.prediction, binary)
        fidelity = scalar_metrics(g.teacher, g.prediction)
        metrics.append(
            {
                "query": query,
                "method": method,
                "authors": len(g),
                "teacher_r": fidelity["pearson_r"],
                "teacher_mse": fidelity["rmse"] ** 2,
                "external_r": external["pearson_r"],
                "external_auc": external.get("auc"),
            }
        )
        moments.append(
            {
                "query": query,
                "method": method,
                **covariance_split(g.y_true, g.teacher, g.prediction),
            }
        )
        for control in ("teacher", "e5_large"):
            intervals.append(
                {
                    "query": query,
                    "method": method,
                    "control": control,
                    **paired_intervals(g.y_true, g.prediction, g[control], binary),
                }
            )
    for filename, records in (
        ("metrics", metrics),
        ("covariance_split", moments),
        ("paired_intervals", intervals),
    ):
        pd.DataFrame(records).to_csv(out / f"{filename}.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "oof_sha256": sha(path),
        "manifest_sha256": sha(out / "manifest.json"),
        "wall_seconds": time.monotonic() - start,
        "new_neural_forwards": 0,
        "questionnaire_fit": False,
        "metrics": metrics,
        "interval_scope": "paired authors conditional on frozen OOF fits, not repeated training",
    }
    write(out / "summary.json", summary)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    return frame


def main():
    """Run the finite supervised-coordinate contrast with a ten-minute fit cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private artifacts belong in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_models").mkdir()
    start = time.monotonic()
    source = args.connection / "private_predictions.csv"
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S5A_SUPERVISED_COORDINATES_V1",
            "script_sha256": sha(__file__),
            "source_sha256": sha(source),
            "methods": METHODS,
            "ranks": RANKS,
            "alphas": ALPHAS,
            "scope": "exposed D1 source authors; no new external confirmation",
            "selection": "inner3 mean teacher MSE; outer5 s4_outer; no questionnaire fit",
            "time_cap_seconds": 600,
            "new_neural_forwards": 0,
        },
    )
    ids, features = feature_cache(args.connection, args.panel)
    targets = pd.read_csv(
        source, dtype={"author_id": str}, usecols=["author_id", "query", "native_query"]
    )
    rows, selections = [], []
    with threadpool_limits(limits=2):
        for query, matrices in features.items():
            names, x, teacher = teacher_arrays(ids, matrices["mean"], targets, query)
            folds = author_folds(names, 5, "s4_outer")
            for fold in range(5):
                train = np.array(
                    sorted(
                        np.flatnonzero(folds != fold),
                        key=lambda j: key(names[j], f"s4_{fold}"),
                    )
                )
                test = np.flatnonzero(folds == fold)
                candidates = []
                predictions = {}
                for method in METHODS:
                    rank, alpha, loss = inner_select(
                        x[train],
                        teacher[train],
                        names[train],
                        method,
                        f"s4_inner_{fold}",
                    )
                    coef, intercept = fit_student(
                        x[train], teacher[train], method, rank, alpha
                    )
                    predictions[method] = x[test] @ coef + intercept
                    np.savez_compressed(
                        out / "private_models" / f"{query}_{method}_{fold}.npz",
                        coef=coef,
                        intercept=intercept,
                    )
                    candidates.append((loss, method))
                    selections.append(
                        {
                            "query": query,
                            "fold": fold,
                            "method": method,
                            "rank": rank,
                            "alpha": alpha,
                            "inner_mse": loss,
                        }
                    )
                chosen = min(candidates)[1]
                predictions["inner_chosen"] = predictions[chosen]
                selections.append(
                    {
                        "query": query,
                        "fold": fold,
                        "method": "inner_chosen",
                        "chosen": chosen,
                    }
                )
                for method, pred in predictions.items():
                    rows.extend(
                        {
                            "author_id": names[j],
                            "query": query,
                            "method": method,
                            "fold": fold,
                            "teacher": float(teacher[j]),
                            "prediction": float(p),
                        }
                        for j, p in zip(test, pred, strict=True)
                    )
                print(f"S5A {query} fold {fold + 1}/5, selected {chosen}", flush=True)
                if time.monotonic() - start > 600:
                    pd.DataFrame(rows).to_csv(out / "private_partial.csv", index=False)
                    raise TimeoutError("S5A exceeded its ten-minute fit cap")
    freeze_and_report(out, rows, selections, source, start)


if __name__ == "__main__":
    main()

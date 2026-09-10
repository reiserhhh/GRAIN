#!/usr/bin/env python3
"""Source-span ceiling: distinguish missing representation from Ridge shrinkage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.content_coefficients import (
    author_weights,
    raw_design,
)
from scripts.analyze_grain_content_coefficients import verify_predictions
from scripts.run_grain_content_coefficients import ARMS, NATIVE
from scripts.run_grain_real_author_path import BASE, read, sha, write


def span_account(x, y, w, coef):
    """Project source targets, returning a fit ceiling, not new held predictions."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or w.shape != (len(x),)
        or np.any(w <= 0)
    ):
        raise ValueError("Aligned positive source weights required")
    xw, yw = x * np.sqrt(w[:, None]), y * np.sqrt(w[:, None])
    u, s, _ = np.linalg.svd(xw, full_matrices=False)
    rank = int(np.sum(s > max(xw.shape) * np.finfo(float).eps * s[0]))
    projected = u[:, :rank] @ (u[:, :rank].T @ yw)
    ridge = xw @ coef
    energy = float(np.sum(yw**2))
    if energy == 0:
        raise ValueError("Source coefficient energy must be nonzero")
    outside = float(np.sum((yw - projected) ** 2) / energy)
    shrink = float(np.sum((projected - ridge) ** 2) / energy)
    actual = float(np.sum((yw - ridge) ** 2) / energy)
    if abs(actual - outside - shrink) > 1e-10:
        raise ValueError("Source projection identity failed")
    return {
        "source_span_rank": rank,
        "best_span_error": outside,
        "ridge_penalty_error": shrink,
        "source_ridge_error": actual,
        "identity_error": actual - outside - shrink,
    }


def analyze(out, name):
    """Inspect fixed source designs only; do not score an OLS model on held users."""
    m = verify_predictions(out)
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "new_predictors": 0,
            "source_span_projections": 24,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    indices = [r["index"] for r in read(out / "private_membership.json")["authors"]]
    packets = {}
    for i in indices:
        with np.load(out / f"private_inputs_{i}.npz") as z:
            packets[i] = (z["hidden"], z["grain"])
    rows = []
    with threadpool_limits(limits=4):
        for q in m["queries"]:
            for held in indices:
                src = [i for i in indices if i != held]
                for arm in ARMS:
                    with np.load(out / q / f"private_model_{held}_{arm}.npz") as z:
                        model = dict(z)
                    xx, yy = [], []
                    full_energy = 0.0
                    for i in src:
                        h, g = packets[i]
                        x = raw_design(h, g, model, bool(model["with_grain"]))
                        xx.append(
                            np.column_stack(
                                [
                                    np.ones(len(h)),
                                    (x - model["feature_center"])
                                    / model["feature_scale"],
                                ]
                            )
                        )
                        with np.load(
                            NATIVE / "private_packets" / f"a{i:02d}_{q}.npz"
                        ) as z:
                            j = z["body_gradient"].astype(float)
                        yy.append(len(j) * j @ model["basis"].T)
                        full_energy += np.sum((h - model["h_center"]) ** 2) / (
                            3 * len(h)
                        )
                    account = span_account(
                        np.concatenate(xx),
                        np.concatenate(yy),
                        author_weights([len(x) for x in xx]),
                        model["weights"],
                    )
                    rows.append(
                        {
                            "query": q,
                            "arm": arm,
                            "held": held,
                            "source_hidden_pca_energy": float(
                                np.sum(model["h_singular"] ** 2) / full_energy
                            ),
                            **account,
                        }
                    )
    d = pd.DataFrame(rows)
    d.to_csv(dest / "private_span.csv", index=False)
    g = (
        d.groupby(["query", "arm"], sort=True)
        .agg(
            folds=("held", "size"),
            source_hidden_pca_energy=("source_hidden_pca_energy", "mean"),
            span_rank_min=("source_span_rank", "min"),
            span_rank_max=("source_span_rank", "max"),
            best_span_error=("best_span_error", "mean"),
            ridge_penalty_error=("ridge_penalty_error", "mean"),
            source_ridge_error=("source_ridge_error", "mean"),
        )
        .reset_index()
    )
    g.to_csv(dest / "source_span.csv", index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "source_span_projections": 24,
            "new_predictors": 0,
            "new_neural_calls": 0,
            "identity_error_max": float(d.identity_error.abs().max()),
            "files": {
                p: sha(dest / p) for p in ("private_span.csv", "source_span.csv")
            },
        },
    )
    print(g.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_content_coefficients_v1")
    p.add_argument("--name", default="span_analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.name)

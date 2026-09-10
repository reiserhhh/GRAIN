#!/usr/bin/env python3
"""Keep energy, local action and source-disjoint reading as separate outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_grain_query_baseline import corr
from scripts.run_grain_position_reader import verified
from scripts.run_grain_real_author_path import BASE, read, sha, write

KEYS = ["query", "method", "rank", "access"]


def action_metrics(actual, predicted):
    """Descriptive tangent fidelity; observations are not independent people."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.ndim != 1 or y.shape != p.shape or len(y) < 3 or not np.isfinite([y, p]).all():
        raise ValueError("Matched finite tangent arrays required")
    energy = np.sum(y * y)
    return {
        "action_r": corr(y, p),
        "action_mse_over_zero": float(np.sum((y - p) ** 2) / energy)
        if energy
        else float("nan"),
        "action_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "actual_action_rms": float(np.sqrt(np.mean(y * y))),
        "predicted_action_rms": float(np.sqrt(np.mean(p * p))),
        "same_sign": int(np.sum(y * p > 0)),
        "nonzero_targets": int(np.count_nonzero(y)),
        "actions": len(y),
    }


def analyze(out, name):
    """Summarize immutable numerical field/action records, without choosing ranks."""
    m = verified(out)
    freeze = read(out / "prediction_freeze.json")
    if sha(out / "manifest.json") != freeze["manifest_sha256"]:
        raise ValueError("Construction manifest changed")
    for p, digest in freeze["files"].items():
        if sha(out / p) != digest:
            raise ValueError("Frozen reader output changed")
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "new_neural_calls": 0,
            "human_labels_read": False,
            "intervals": "No population significance claims from4 reused authors",
        },
    )
    fields = pd.concat(
        [pd.read_csv(out / q / "private_fields.csv") for q in m["queries"]],
        ignore_index=True,
    )
    actions = pd.concat(
        [pd.read_csv(out / q / "private_actions.csv") for q in m["queries"]],
        ignore_index=True,
    )
    if (
        len(fields) != 132
        or len(actions) != 528
        or fields.duplicated(KEYS + ["index"]).any()
        or actions.duplicated(KEYS + ["index", "factor"]).any()
    ):
        raise ValueError("Complete4-author x3-query x11-reader comparison required")
    comparisons = []
    by_factor = []
    for key, g in actions.groupby(KEYS, sort=True):
        meta = dict(zip(KEYS, key, strict=True))
        h = fields
        for k, v in meta.items():
            h = h[h[k] == v]
        if len(h) != 4 or g["index"].nunique() != 4 or len(g) != 16:
            raise ValueError("Four authors and four factors required per comparison")
        row = meta | {
            "authors": 4,
            "gradient_error_ratio_mean": float(h.field_error_ratio.mean()),
            "gradient_error_ratio_min": float(h.field_error_ratio.min()),
            "gradient_error_ratio_max": float(h.field_error_ratio.max()),
            "gradient_error_ratio_pooled": float(
                h.field_error_energy.sum() / h.field_actual_energy.sum()
            ),
            **action_metrics(g.actual, g.predicted),
        }
        comparisons.append(row)
        for factor, a in g.groupby("factor", sort=True):
            by_factor.append(
                meta
                | {
                    "factor": factor,
                    "authors": 4,
                    **action_metrics(a.actual, a.predicted),
                }
            )
    # A response error bound is exact but often loose when signal cancellations occur.
    discrepancy = np.maximum(
        abs(actions.predicted - actions.actual) - actions.error_bound, 0
    )
    if discrepancy.max() > 1e-10:
        raise ValueError("Cauchy action bound violated")
    pd.DataFrame(comparisons).to_csv(dest / "reader_comparison.csv", index=False)
    pd.DataFrame(by_factor).to_csv(dest / "factor_actions.csv", index=False)
    counts = (
        fields.groupby("query")
        .agg(
            authors=("index", "nunique"),
            tokens_min=("tokens", "min"),
            tokens_max=("tokens", "max"),
            field_rows=("index", "size"),
        )
        .reset_index()
    )
    counts.to_csv(dest / "coverage.csv", index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "authors": 4,
            "queries": 3,
            "fields": 12,
            "unique_original_tangents": 48,
            "reader_comparisons": len(comparisons),
            "factor_comparisons": len(by_factor),
            "new_neural_calls": 0,
            "human_labels_read": False,
            "cauchy_violation_max": float(discrepancy.max()),
            "files": {
                p: sha(dest / p)
                for p in ("reader_comparison.csv", "factor_actions.csv", "coverage.csv")
            },
        },
    )
    print(
        {"status": "ANALYSIS_COMPLETE", "comparisons": len(comparisons), "authors": 4}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_position_reader_v1")
    p.add_argument("--name", default="analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.name)

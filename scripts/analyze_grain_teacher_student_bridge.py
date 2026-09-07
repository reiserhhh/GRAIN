#!/usr/bin/env python3
"""Paired author-level inference and covariance decomposition for frozen S4 OOF scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_multiquery_sensitivity import corr
from scripts.run_grain_real_author_path import BASE, read, sha, write


def covariance_split(y, teacher, student):
    """Exactly decompose label covariance; prediction correlation need not be conserved."""
    y, teacher, student = [np.asarray(v, float) for v in (y, teacher, student)]
    cov = lambda a, b: float(np.mean((a - a.mean()) * (b - b.mean())))
    residual = teacher - student
    return {
        "teacher_label_covariance": cov(y, teacher),
        "student_label_covariance": cov(y, student),
        "residual_label_covariance": cov(y, residual),
        "identity_error": abs(cov(y, teacher) - cov(y, student) - cov(y, residual)),
        "teacher_sd": float(teacher.std()),
        "student_sd": float(student.std()),
        "residual_label_r": corr(y, residual),
        "teacher_student_r": corr(teacher, student),
    }


def paired_intervals(y, student, control, binary, draws=1000):
    """Resample paired authors conditional on already frozen cross-fit predictions."""
    y, student, control = [np.asarray(v, float) for v in (y, student, control)]
    metric = roc_auc_score if binary else corr
    rng = np.random.default_rng(20260908)
    estimates, differences = [], []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        if np.ptp(y[ix]) == 0:
            continue
        a, b = metric(y[ix], student[ix]), metric(y[ix], control[ix])
        if a is not None and b is not None:
            estimates.append(a)
            differences.append(a - b)
    delta = float(metric(y, student) - metric(y, control))
    return {
        "student_lower": float(np.quantile(estimates, 0.025)),
        "student_upper": float(np.quantile(estimates, 0.975)),
        "delta": delta,
        "delta_lower": float(np.quantile(differences, 0.025)),
        "delta_upper": float(np.quantile(differences, 0.975)),
    }


def main():
    """Analyze all-sample mean/query candidates without retuning either student."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private bridge analysis must remain in results")
    out.mkdir(parents=True, exist_ok=False)
    path = args.curve / "private_oof_before_outcomes.csv"
    summary = read(args.curve / "summary.json")
    if sha(path) != summary["oof_sha256"]:
        raise ValueError("Frozen OOF predictions changed")
    frame = pd.read_csv(path, dtype={"author_id": str, "size": str})
    source = pd.read_csv(
        args.connection / "private_predictions.csv", dtype={"author_id": str}
    )
    source = source.drop(columns="student")
    frame = frame[frame["size"] == "all"].merge(
        source, on=["author_id", "query"], validate="many_to_one"
    )
    if not np.allclose(frame.teacher, frame.native_query):
        raise ValueError("Mismatched teacher scores")
    intervals, moments, metrics = [], [], []
    for (query, family), g in frame.groupby(["query", "family"]):
        moments.append(
            {
                "query": query,
                "family": family,
                **covariance_split(g.y_true, g.teacher, g.prediction),
            }
        )
        metrics.append(
            {
                "query": query,
                "family": family,
                **scalar_metrics(g.y_true, g.prediction, query == "TF_cont"),
            }
        )
        for control in ("native_query", "e5_large"):
            intervals.append(
                {
                    "query": query,
                    "family": family,
                    "control": control,
                    "metric": "auc" if query == "TF_cont" else "pearson_r",
                    **paired_intervals(
                        g.y_true, g.prediction, g[control], query == "TF_cont"
                    ),
                }
            )
    for name, rows in (
        ("paired_intervals", intervals),
        ("covariance_split", moments),
        ("student_metrics", metrics),
    ):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "new_fits": 0,
            "new_neural_forwards": 0,
            "script_sha256": sha(__file__),
            "interval_scope": "paired authors, conditional on frozen OOF fits; not repeated-training uncertainty",
            "paired_intervals": intervals,
            "covariance_split": moments,
            "metrics": metrics,
        },
    )
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)
    print(pd.DataFrame(moments).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

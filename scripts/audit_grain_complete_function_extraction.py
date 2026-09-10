#!/usr/bin/env python3
"""Cache-only NEXT57 audit: source exclusion, objective equations and exact replay."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.complete_function_extraction import design
from scripts.run_grain_complete_function_extraction import (
    BASE,
    OBJECTIVES,
    QUERIES,
    apply_scalar,
    arrays,
    basis_path,
    feature_packet,
    frozen,
    model_path,
    read,
    ref,
    selected,
    sha,
    source_jobs,
    training_arrays,
    verify,
    write,
)


def audit(out):
    """Verify equations without fitting again or requesting any native model output."""
    start = time.process_time()
    verify(out)
    fm, fp, fr = [frozen(out, d) for d in ("models", "predictions", "replay")]
    jobs = read(out / "private_jobs.json")
    errors = {
        "normal_equation": 0.0,
        "source_normalization": 0.0,
        "derivative_finite_difference": 0.0,
        "metric": 0.0,
    }
    for reference in sorted({ref(j) for j in jobs}):
        src = source_jobs(jobs, reference)
        b = selected(basis_path(out, reference), ("mean", "basis", "members"))
        np.testing.assert_array_equal(b["members"], [j["index"] for j in src])
        for qi, q in enumerate(QUERIES):
            x, d, y, t = training_arrays(out, src, b, qi)
            # Match the estimator's explicit float64 source reduction, not NPZ dtype.
            y = y.astype(float)
            scale = x.std(axis=0)
            scale[scale < 1e-8] = 1
            expected = {
                "center": x.mean(axis=0),
                "scale": scale,
                "origin": y.mean(),
                "target_scale": max(float(y.std()), 1e-6),
            }
            for o in OBJECTIVES:
                m = arrays(model_path(out, reference, q, o))
                np.testing.assert_array_equal(m["members"], b["members"])
                for n, v in expected.items():
                    delta = float(np.max(np.abs(m[n] - v)))
                    errors["source_normalization"] = max(
                        errors["source_normalization"], delta
                    )
                a, target, *_ = design(x, d, y, t, m)
                errors["normal_equation"] = max(
                    errors["normal_equation"],
                    float(np.max(np.abs(a.T @ (a @ m["coef"] - target)))),
                )
    for j in jobs:
        b = selected(basis_path(out, ref(j)), ("mean", "basis"))
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            original, replay = [
                arrays(out / f / (key + ".npz")) for f in ("predictions", "replay")
            ]
            for n in ("score", "tau"):
                np.testing.assert_array_equal(original[n], replay[n])
            phi, dp = feature_packet(out, key, b)
            for oi, o in enumerate(OBJECTIVES):
                m = arrays(model_path(out, ref(j), q, o))
                yp, tp = apply_scalar(phi, dp, m)
                np.testing.assert_array_equal(yp, original["score"][oi])
                np.testing.assert_array_equal(tp, original["tau"][oi])
                for f in range(4):
                    a, c = [
                        apply_scalar(phi + s * 1e-5 * dp[f], dp, m)[0] for s in (1, -1)
                    ]
                    errors["derivative_finite_difference"] = max(
                        errors["derivative_finite_difference"],
                        abs(float((a - c) / 2e-5 - tp[f])),
                    )
    csvs = [
        p
        for p in sorted((out / "analysis").glob("*.csv"))
        if not p.name.startswith(".")
    ]
    for p in csvs:
        if sha(p) != sha(out / "replay_analysis" / p.name):
            raise ValueError("Analysis replay differs: " + p.name)
    # Reconstruct core metrics without using the report's fidelity helper.
    for n, fields in (
        ("scores", ("score_r", "score_mse_over_source_constant")),
        ("actions", ("action_r", "action_mse_over_zero")),
    ):
        raw = pd.read_csv(out / "analysis" / ("private_" + n + ".csv"))
        summary = pd.read_csv(out / "analysis" / (n + ".csv"))
        for row in summary.to_dict("records"):
            g = raw[
                (raw.cohort == row["cohort"])
                & (raw.query_name == row["query_name"])
                & (raw.candidate == row["candidate"])
            ]
            actual, pred = g.actual.to_numpy(), g.predicted.to_numpy()
            denominator = (
                np.sum((actual - g.constant.to_numpy()) ** 2)
                if n == "scores"
                else np.sum(actual**2)
            )
            vals = [
                np.corrcoef(actual, pred)[0, 1],
                np.sum((actual - pred) ** 2) / denominator,
            ]
            errors["metric"] = max(
                errors["metric"],
                *[abs(float(v - row[f])) for f, v in zip(fields, vals, strict=True)],
            )
    statuses = [
        read(out / "models" / (r + "_status.json"))
        for r in sorted({ref(j) for j in jobs})
    ]
    assert len(statuses) == 5 and sum(s["fits"] for s in statuses) == 20
    assert all(s["extra_denials"] == s["current_target_reads"] == 0 for s in statuses)
    assert fp["extra_denials"] == fr["extra_denials"] == 0
    assert errors["source_normalization"] == 0
    assert errors["normal_equation"] < 1e-8 and errors["metric"] < 1e-8
    assert errors["derivative_finite_difference"] < 1e-7
    cpu = time.process_time() - start
    result = {
        "status": "PASS",
        "errors": errors,
        "fits": 0,
        "new_neural_calls": 0,
        "contexts_replayed": 24,
        "same_function_checks": 192,
        "csvs_replayed": len(csvs),
        "cpu_seconds": cpu,
        "cumulative_compute_cpu_seconds": fm["cpu_seconds"]
        + fp["cpu_seconds"]
        + fr["cpu_seconds"]
        + cpu,
        "scope": "Native fixed-model outputs, not human questionnaire validity",
    }
    assert result["cumulative_compute_cpu_seconds"] < 180
    write(out / "replay_audit.json", result)
    print(result)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        audit(BASE / "logic_complete_function_extraction_v1")

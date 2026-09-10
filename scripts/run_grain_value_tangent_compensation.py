#!/usr/bin/env python3
"""NEXT58: one cached compensation/source-support account, never a new predictor."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.value_tangent_compensation import (
    author_centered,
    compensation_summary,
    coverage,
    row_space,
    score_parts,
)
from scripts.run_grain_complete_function_extraction import (
    BASE,
    ENDPOINT,
    OBJECTIVES,
    QUERIES,
    apply_scalar,
    arrays,
    basis_path,
    error_account,
    feature_packet,
    fidelity,
    frozen,
    model_path,
    read,
    ref,
    score_metrics,
    selected,
    sha,
    source_jobs,
    verify,
    verify_hashes,
    write,
)

OLD = BASE / "logic_complete_function_extraction_v1"
PLAN = ROOT / "docs/GRAIN_VALUE_TANGENT_COMPENSATION_PLAN_V1.md"
FACTORS = [f"whole:F{i:02d}" for i in range(1, 5)]


def prepare(out):
    """Freeze the existing panel, functions and analysis code; do not learn anything."""
    m = verify(OLD)
    fm, fp = frozen(OLD, "models"), frozen(OLD, "predictions")
    inputs = dict(m["inputs"])
    paths = [OLD / n for n in m["prepared"]]
    paths += [OLD / "models" / n for n in fm["files"]]
    paths += [OLD / "predictions" / n for n in fp["files"]]
    paths += [
        OLD / "manifest.json",
        OLD / "analysis/summary.json",
        OLD / "replay_audit.json",
        OLD / "models/freeze.json",
        OLD / "predictions/freeze.json",
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/value_tangent_compensation.py",
        ROOT / "tests/test_grain_value_tangent_compensation.py",
    ]
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    inputs.update(m["evaluation"])
    out.mkdir(exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_VALUE_TANGENT_COMPENSATION_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": inputs,
            "max_cpu_seconds": 60,
            "cohorts": ["source_fit", "source4", "new8"],
            "fits": 0,
            "neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", "current_materials": 12, "infit_source_views": 4})


def analyze(out, folder):
    """Inspect source-fit and two application groups using the exact frozen models."""
    start = time.process_time()
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    jobs = read(OLD / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    models, spaces, bases, source_diagnostics = {}, {}, {}, []
    for reference in sorted({ref(j) for j in jobs}):
        b = selected(basis_path(OLD, reference), ("mean", "basis", "members"))
        bases[reference] = b
        src = source_jobs(jobs, reference)
        np.testing.assert_array_equal(b["members"], [j["index"] for j in src])
        for qi, q in enumerate(QUERIES):
            pair = [arrays(model_path(OLD, reference, q, o)) for o in OBJECTIVES]
            models[reference, q] = pair
            features = [feature_packet(OLD, j["query_keys"][qi], b) for j in src]
            phi, dp = [np.asarray(v) for v in zip(*features, strict=True)]
            basis, diag = row_space(phi, dp, pair[0])
            if diag["rank"] != (20 if reference == "all4" else 15):
                raise ValueError("Source data rank differs from NEXT57")
            spaces[reference, q] = basis
            source_diagnostics.append({"reference": reference, "query_name": q, **diag})
    score_rows, action_rows, pieces_rows, coverage_rows = [], [], [], []
    checks = {
        "score_sum_error": 0.0,
        "derivative_error": 0.0,
        "cached_prediction_error": 0.0,
    }
    # In-fit source4 is an extra view of existing authors, never extra subjects.
    views = [(j, j["cohort"], ref(j)) for j in jobs]
    views += [(j, "source_fit", "all4") for j in jobs if j["cohort"] == "source4"]
    for j, cohort, reference in views:
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            v, joint = models[reference, q]
            phi, dp = feature_packet(OLD, key, bases[reference])
            yp, tp = apply_scalar(phi, dp, v)
            yj, tj = apply_scalar(phi, dp, joint)
            pieces, dt = score_parts(phi, dp, v, joint)
            checks["score_sum_error"] = max(
                checks["score_sum_error"], abs(float(pieces.sum() - (yj - yp)))
            )
            checks["derivative_error"] = max(
                checks["derivative_error"], float(np.max(np.abs(dt - (tj - tp))))
            )
            if cohort != "source_fit":
                p = arrays(OLD / "predictions" / (key + ".npz"))
                delta = max(
                    float(np.max(np.abs(p["score"] - [yp, yj]))),
                    float(np.max(np.abs(p["tau"] - [tp, tj]))),
                )
                checks["cached_prediction_error"] = max(
                    checks["cached_prediction_error"], delta
                )
            native_folder = (
                "capture_source" if j["cohort"] == "source4" else "capture_new"
            )
            target = selected(
                ENDPOINT / native_folder / (key + ".npz"), ("score", "tau")
            )
            meta = {"author_key": j["key"], "cohort": cohort, "query_name": q}
            pieces_rows.append(
                meta
                | {
                    "intercept": pieces[0],
                    "body": pieces[1],
                    "query": pieces[2],
                    "delta_score": yj - yp,
                    "value_error": yp - float(target["score"]),
                    "joint_error": yj - float(target["score"]),
                }
            )
            for o, score, tau in zip(OBJECTIVES, (yp, yj), (tp, tj), strict=True):
                score_rows.append(
                    meta
                    | {
                        "candidate": o,
                        "actual": float(target["score"]),
                        "predicted": score,
                        "constant": float(v["origin"]),
                    }
                )
                action_rows += [
                    meta
                    | {
                        "candidate": o,
                        "factor": f,
                        "actual": float(target["tau"][i]),
                        "predicted": tau[i],
                    }
                    for i, f in enumerate(FACTORS)
                ]
            z, dz = (phi - v["center"]) / v["scale"], dp / v["scale"]
            rows = np.vstack((np.r_[1.0, z], np.column_stack((np.zeros(4), dz))))
            cov = coverage(rows, spaces[reference, q])
            coverage_rows += [
                meta
                | {
                    "row_type": "score" if i == 0 else "direction",
                    "factor": f,
                    "coverage": cov[i],
                }
                for i, f in enumerate(["score", *FACTORS])
            ]
    sd, ad, parts, cov = [
        pd.DataFrame(r) for r in (score_rows, action_rows, pieces_rows, coverage_rows)
    ]
    if (len(sd), len(ad), len(parts), len(cov)) != (64, 256, 32, 160):
        raise ValueError("Declared source/application inventory differs")
    if max(checks.values()) > 1e-10:
        raise ValueError("Frozen function identity differs")
    tables = {
        "private_scores": sd,
        "private_actions": ad,
        "private_parts": parts,
        "private_coverage": cov,
        "source_design": pd.DataFrame(source_diagnostics),
    }
    comp, cover, sm, errors = [], [], [], []
    group = ["cohort", "query_name"]
    for k, g in parts.groupby(group, sort=True):
        comp.append(
            dict(zip(group, k, strict=True))
            | compensation_summary(g[["intercept", "body", "query"]], g.value_error)
        )
    for k, g in cov.groupby(group + ["row_type", "factor"], sort=True):
        a = g.coverage.to_numpy()
        good = a[np.isfinite(a)]
        cover.append(
            dict(zip(group + ["row_type", "factor"], k, strict=True))
            | {
                "authors": g.author_key.nunique(),
                "undefined": int(len(a) - len(good)),
                "mean_coverage": float(good.mean()) if len(good) else np.nan,
                "min_coverage": float(good.min()) if len(good) else np.nan,
                "max_coverage": float(good.max()) if len(good) else np.nan,
            }
        )
    for k, g in sd.groupby(group + ["candidate"], sort=True):
        sm.append(
            dict(zip(group + ["candidate"], k, strict=True))
            | score_metrics(g.actual, g.predicted, g.constant)
        )
    for kind, df in (("score", sd), ("response", ad)):
        for k, g in df.groupby(group, sort=True):
            index = ["author_key"] + (["factor"] if kind == "response" else [])
            a, b = [
                g[g.candidate == o].set_index(index).sort_index() for o in OBJECTIVES
            ]
            if not a.index.equals(b.index):
                raise ValueError("Error pair membership differs")
            np.testing.assert_array_equal(a.actual, b.actual)
            errors.append(
                dict(zip(group, k, strict=True))
                | {"kind": kind}
                | error_account(a.actual, a.predicted, b.predicted)
            )
    fr, pooled = [], []
    for k, g in ad.groupby(group + ["candidate"], sort=True):
        actual, predicted = [
            g.pivot(index="author_key", columns="factor", values=n)
            .loc[:, FACTORS]
            .sort_index()
            .to_numpy()
            for n in ("actual", "predicted")
        ]
        for mode in ("raw", "author_centered"):
            y, p = (
                (actual, predicted)
                if mode == "raw"
                else (author_centered(actual), author_centered(predicted))
            )
            meta = dict(zip(group + ["candidate"], k, strict=True)) | {
                "mode": mode,
                "authors": len(y),
            }
            pooled.append(meta | fidelity(y.ravel(), p.ravel()))
            fr += [
                meta | {"factor": f} | fidelity(y[:, i], p[:, i])
                for i, f in enumerate(FACTORS)
            ]
    tables.update(
        compensation=pd.DataFrame(comp),
        coverage=pd.DataFrame(cover),
        scores=pd.DataFrame(sm),
        error_changes=pd.DataFrame(errors),
        factor_responses=pd.DataFrame(fr),
        pooled_responses=pd.DataFrame(pooled),
    )
    for n, table in tables.items():
        table.to_csv(dst / (n + ".csv"), index=False)
    cpu = time.process_time() - start
    if cpu > m["max_cpu_seconds"]:
        raise TimeoutError("Cache-only CPU cap exceeded")
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "fits": 0,
            "neural_calls": 0,
            "human_labels_read": False,
            "unique_authors": 12,
            "material_views": 16,
            "checks": checks,
            "files": {n + ".csv": sha(dst / (n + ".csv")) for n in tables},
        },
    )
    print(
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "checks": checks,
            "tables": len(tables),
        }
    )


def audit(out):
    """Verify exact replay and independent covariance/coverage calculations."""
    start = time.process_time()
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    a, b = [read(out / n / "summary.json") for n in ("analysis", "replay")]
    if (
        a["status"] != b["status"]
        or a["status"] != "COMPLETE"
        or a["files"] != b["files"]
    ):
        raise ValueError("One cached replay must reproduce every CSV exactly")
    for folder, s in (("analysis", a), ("replay", b)):
        verify_hashes(out / folder, s["files"])
    raw = pd.read_csv(out / "analysis/private_parts.csv")
    summary = pd.read_csv(out / "analysis/compensation.csv")
    max_error = 0.0
    for r in summary.to_dict("records"):
        g = raw[(raw.cohort == r["cohort"]) & (raw.query_name == r["query_name"])]
        v = g[["body", "query"]].to_numpy()
        c = np.cov(v.T, bias=True)
        expected = c.sum() / np.trace(c)
        max_error = max(
            max_error,
            abs(float(expected - r["cancellation_ratio"])),
            abs(float(np.mean(g.joint_error**2 - g.value_error**2) - r["mse_change"])),
        )
    # Check SVD coverage against a Moore-Penrose projection, not the runner helper.
    jobs = read(OLD / "private_jobs.json")
    max_projection_error = 0.0
    for reference in sorted({ref(j) for j in jobs}):
        bs = selected(basis_path(OLD, reference), ("mean", "basis"))
        src = source_jobs(jobs, reference)
        for qi, q in enumerate(QUERIES):
            model = arrays(model_path(OLD, reference, q, "value"))
            xp, dp = [
                np.asarray(v)
                for v in zip(
                    *[feature_packet(OLD, j["query_keys"][qi], bs) for j in src],
                    strict=True,
                )
            ]
            z, dz = (xp - model["center"]) / model["scale"], dp / model["scale"]
            matrix = np.vstack(
                (
                    np.column_stack((np.ones(len(z)), z)) / np.sqrt(len(z)),
                    np.column_stack((np.zeros(len(z) * 4), dz.reshape(-1, 64)))
                    / np.sqrt(len(z) * 4),
                )
            )
            v, _ = row_space(xp, dp, model)
            proj = (
                np.linalg.pinv(matrix, rcond=max(matrix.shape) * np.finfo(float).eps)
                @ matrix
            )
            max_projection_error = max(
                max_projection_error, float(np.max(np.abs(proj - v.T @ v)))
            )
    if max_error > 1e-9 or max_projection_error > 1e-9:
        raise ValueError("Independent covariance or projection identity failed")
    cpu = time.process_time() - start
    if a["cpu_seconds"] + b["cpu_seconds"] + cpu > m["max_cpu_seconds"]:
        raise TimeoutError("Total cached analysis budget exceeded")
    result = {
        "status": "PASS",
        "csvs_replayed": len(a["files"]),
        "account_error": max_error,
        "projection_matrix_error": max_projection_error,
        "cpu_seconds": cpu,
        "total_measured_cpu_seconds": a["cpu_seconds"] + b["cpu_seconds"] + cpu,
        "fits": 0,
        "neural_calls": 0,
        "human_labels_read": False,
    }
    write(out / "replay_audit.json", result)
    print(result)


def main():
    """Separate pre-outcome freezing, analysis and one exact replay."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "analyze", "audit"), required=True)
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_value_tangent_compensation_v1"
    )
    p.add_argument("--folder", choices=("analysis", "replay"), default="analysis")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "analyze":
            analyze(a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()

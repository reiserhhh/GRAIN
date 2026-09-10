#!/usr/bin/env python3
"""Publish NEXT68 aggregates and a post-outcome positional-range account."""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_coupled_pooling import (
    ARMS,
    OUT,
    manifest,
    model_path,
    old,
    pure_input,
    teacher_paths,
)


def positional_account(target, prediction, positions):
    """Relax shared coefficients: project a B response onto constant/position fields.

    This is a per-observation descriptive upper bound, not a fitted predictor.
    Projection coefficients are discarded and never applied to another author.
    """
    target, prediction, positions = (
        np.asarray(a, float) for a in (target, prediction, positions)
    )
    if (
        target.ndim != 2
        or prediction.shape != target.shape
        or positions.shape != (len(target),)
        or len(target) < 2
        or not all(np.isfinite(x).all() for x in (target, prediction, positions))
    ):
        raise ValueError("Aligned finite B fields and positions required")
    x = np.column_stack((np.ones(len(target)), positions))
    if np.linalg.matrix_rank(x) != 2:
        raise ValueError("Two distinct positions required")
    projected = x @ np.linalg.lstsq(x, target, rcond=None)[0]
    prediction_projection = x @ np.linalg.lstsq(x, prediction, rcond=None)[0]
    energy = np.mean(target**2)
    if energy <= 0:
        raise ValueError("Nonzero target response required")
    kept, residual = np.mean(projected**2), np.mean((target - projected) ** 2)
    return {
        "target_energy": float(energy),
        "representable_energy": float(kept),
        "residual_energy": float(residual),
        "prediction_error": float(np.mean((prediction - target) ** 2)),
        "prediction_off_subspace_rms": float(
            np.sqrt(np.mean((prediction - prediction_projection) ** 2))
        ),
        "orthogonality_error": float(abs(energy - kept - residual)),
    }


def publish():
    """Publish primary aggregates and a separate source-only range restriction."""
    start = time.process_time()
    m = manifest(OUT)
    audit = old.read(OUT / "audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Primary audit required")
    old.verify_hashes(OUT / "analysis", summary["files"])
    old.frozen(OUT, "models")
    old.frozen(OUT, "predictions")
    dst = ROOT / "reports/grain_vector_mechanisms/coupled_pooling_v1"
    dst.mkdir(exist_ok=True)
    hashes = {}
    for name, digest in summary["files"].items():
        if name.startswith("private_"):
            continue
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {"author_key", "text", "user_id"}:
            raise ValueError("Private field in public aggregate")
        shutil.copyfile(src, dst / name)
        hashes[name] = digest
    jobs = [j for j in old.read(OUT / "private_jobs.json") if j["cohort"] == "source4"]
    rows = []
    for qi, q in enumerate(old.QUERIES):
        model = old.arrays(model_path(OUT, q, ARMS[1]))
        accounts = []
        for j in jobs:
            k = j["query_keys"][qi]
            data = pure_input(j, "original", k)
            n = len(data["positions"])
            cut = n // 2
            ps = [
                OUT / "predictions" / (k + "_" + v + ".npz") for v in ("plus", "minus")
            ]
            gs = [teacher_paths(j, v, k)[0] for v in ("plus", "minus")]
            old.check_paths(m, gs)
            pred = [old.arrays(p)["density"][1] for p in ps]
            grad = [
                old.selected(p, ("body_gradient",))["body_gradient"].astype(float)
                for p in gs
            ]
            target = n * (((grad[0] - grad[1]) / 2) @ model["basis"]) * model["scale"]
            args = (
                target[cut:],
                ((pred[0] - pred[1]) / 2)[cut:],
                np.linspace(-1, 1, n)[cut:],
            )
            result = positional_account(*args)
            if result != positional_account(*args):
                raise ValueError("Post-outcome arithmetic replay changed")
            accounts.append(result)
        means = pd.DataFrame(accounts).mean()
        energy = means.target_energy
        cap = means.representable_energy / energy
        relative_error = means.prediction_error / energy
        rows.append(
            {
                "query_name": q,
                "authors": len(accounts),
                "representable_fraction": cap,
                "unconstrained_relative_sse_floor": means.residual_energy / energy,
                "joint_relative_sse": relative_error,
                "attained_sse_reduction_fraction_of_bound": (1 - relative_error) / cap,
                "max_prediction_off_subspace_rms": max(
                    a["prediction_off_subspace_rms"] for a in accounts
                ),
                "max_orthogonality_error": max(
                    a["orthogonality_error"] for a in accounts
                ),
            }
        )
    table = pd.DataFrame(rows)
    if (
        table.max_prediction_off_subspace_rms.max() > 1e-10
        or table.max_orthogonality_error.max() > 1e-14
    ):
        raise ValueError("Positional range identity failed")
    path = dst / "post_outcome_positional_range.csv"
    table.to_csv(path, index=False)
    hashes[path.name] = old.sha(path)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": hashes,
            "scope": "source4 fitting, previously exposed disjoint new8 application, two native model queries; no human criterion",
            "post_outcome_scope": "source-only per-author positional subspace bound; no predictive coefficients retained or applied",
            "post_outcome_code_sha256": old.sha(Path(__file__)),
            "post_outcome_test_sha256": old.sha(
                ROOT / "tests/test_grain_coupled_pooling_account.py"
            ),
            "post_outcome_cpu_seconds": time.process_time() - start,
            "additional_predictive_fits_or_neural_calls": 0,
        },
    )
    print(
        {"status": "PUBLISHED", "tables": len(hashes), "positional_range": rows},
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        publish()

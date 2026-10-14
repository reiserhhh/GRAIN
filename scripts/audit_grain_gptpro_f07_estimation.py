#!/usr/bin/env python3
"""Independently replay F07 outcomes and likelihoods, retaining optimizer failures."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr


def probabilities(theta, z, row):
    """Rebuild channel law without importing the estimator or channel module."""
    location = theta[0] + theta[1] * z
    if row["ordinal"]:
        return np.diff(ndtr(np.array([-np.inf, -1, -0.3, 0.3, 1, np.inf]) - location))
    p = expit(location)
    c = row["presence"]
    s = row["flip"] + (1 - 2 * row["flip"]) * p
    if row["variant"] != "base":
        s = 0.8 * s + 0.1
    if row["variant"] == "mnar":
        c *= (1 + p) / 2
    return np.array([1 - c, c * s, c * (1 - s)])


def objective(theta, row):
    """Independent exact negative log posterior, excluding count constants."""
    value = 0.5 * (theta[0] ** 2 + (theta[1] - 0.5) ** 2 / 0.25)
    for counts, z in zip(np.asarray(row["counts"]), (-1, 1), strict=True):
        p = probabilities(theta, z, row)
        active = counts > 0
        if (p[active] <= 0).any():
            return np.inf
        value -= np.dot(counts[active], np.log(p[active]))
    return float(value)


def main():
    """Require a complete run; replay counts, objectives, errors, and selected refits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads((args.input / "checkpoint.json").read_text())
    if checkpoint["status"] != "COMPUTED_AUDIT_PENDING":
        raise RuntimeError("incomplete cohort")
    paths = sorted(args.input.glob("author_*.json.gz"))
    if len(paths) != checkpoint["expected_authors"]:
        raise RuntimeError("author count mismatch")
    differences, failed_gradients, refits, hashes = [], [], [], {}
    rows_checked = 0
    for file_index, path in enumerate(paths):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt") as stream:
            packet = json.load(stream)
        truth, uniforms = np.asarray(packet["truth"]), np.asarray(packet["uniforms"])
        for row_index, row in enumerate(packet["rows"]):
            n = row["n"]
            for condition_index, z in enumerate((-1, 1)):
                p = probabilities(truth, z, row)
                u = uniforms[:n][np.arange(n) % 2 == condition_index]
                counts = np.bincount(np.searchsorted(np.cumsum(p), u), minlength=len(p))
                np.testing.assert_array_equal(counts, row["counts"][condition_index])
            estimate = np.asarray(row["estimate"])
            differences.append(abs(objective(estimate, row) - row["objective"]))
            np.testing.assert_allclose(
                (estimate - truth) ** 2, row["squared_error"], atol=1e-12
            )
            covariance = np.asarray(row["covariance"])
            assert np.linalg.eigvalsh(covariance).min() > 0
            np.testing.assert_array_equal(
                np.abs(estimate - truth) <= 1.96 * np.sqrt(np.diag(covariance)),
                row["coverage95"],
            )
            if not row["success"]:
                failed_gradients.append(row["gradient_norm"])
            # Predeclared first three sorted authors/all designs; no truth-based selection.
            if file_index < 3:
                reference = minimize(
                    objective,
                    [0, 0.5],
                    args=(row,),
                    method="Powell",
                    options={"xtol": 1e-10, "ftol": 1e-12, "maxiter": 1000},
                )
                refits.append(
                    {
                        "file": path.name,
                        "row": row_index,
                        "reference_success": bool(reference.success),
                        "objective_gap": float(row["objective"] - reference.fun),
                        "estimate_distance": float(
                            np.linalg.norm(estimate - reference.x)
                        ),
                    }
                )
            rows_checked += 1
    summary = {
        "status": "COUNTS_OBJECTIVES_ERRORS_REPLAYED_OPTIMIZER_FLAGS_RETAINED",
        "authors": len(paths),
        "fit_rows": rows_checked,
        "max_objective_replay_difference": max(differences),
        "optimizer_non_success": len(failed_gradients),
        "failed_gradient_quantiles": np.quantile(
            failed_gradients, [0, 0.5, 0.95, 1]
        ).tolist()
        if failed_gradients
        else [],
        "reference_refits": len(refits),
        "max_absolute_refit_objective_gap": max(
            abs(r["objective_gap"]) for r in refits
        ),
        "scope": "same-agent independent channel algebra/Powell sample; aggregate paired intervals and full covariance replay remain separate",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, data in (
        ("summary.json", summary),
        ("reference_refits.json", refits),
        ("hashes.json", hashes),
    ):
        (args.output / filename).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

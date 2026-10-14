#!/usr/bin/env python3
"""Target-scoped uncertainty and finite calibration contracts for construction."""

import numpy as np
from scipy.stats import norm

from .f13_uncertainty import validate_covariance


def uncertainty_components(names, covariance):
    """Sum a joint error-component covariance without discarding cross terms."""
    c = validate_covariance(covariance)
    if len(names) != len(c) or not names or len(set(names)) != len(names):
        raise ValueError("unique aligned component names required")
    cross = [
        {
            "left": names[i],
            "right": names[j],
            "covariance": float(c[i, j]),
            "contribution_to_total": float(2 * c[i, j]),
        }
        for i in range(len(c))
        for j in range(i + 1, len(c))
    ]
    return {
        "marginal_variances": dict(zip(names, np.diag(c).tolist(), strict=True)),
        "cross_covariances": cross,
        "total_variance": max(0.0, float(c.sum())),
        "component_covariance": c.tolist(),
    }


def calibrate_intervals(truth, lower, upper, unit_ids, scope):
    """Assess frozen 95% intervals; never fit an inflation factor on evaluation."""
    y, lo, hi = (
        np.asarray(truth, float),
        np.asarray(lower, float),
        np.asarray(upper, float),
    )
    if (
        y.ndim != 1
        or not len(y)
        or y.shape != lo.shape
        or y.shape != hi.shape
        or len(unit_ids) != len(y)
    ):
        raise ValueError("aligned nonempty scalar target observations required")
    if len(set(unit_ids)) != len(y):
        raise ValueError("dependent repeated units require a cluster analysis")
    if (
        not all(np.isfinite(x).all() for x in (y, lo, hi))
        or np.any(lo > hi)
        or not scope
    ):
        raise ValueError("finite ordered intervals and explicit scope required")
    n = len(y)
    hits = int(np.count_nonzero((lo <= y) & (y <= hi)))
    p = hits / n
    z = float(norm.ppf(0.975))
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    low, high = float(center - half), float(center + half)
    return {
        "scope": dict(scope),
        "independent_units": n,
        "covered": hits,
        "coverage": p,
        "wilson_95": [low, high],
        "mean_width": float(np.mean(hi - lo)),
        "decision": "PASS" if low <= 0.95 <= high else "FAIL",
        "meaning": "finite coverage check in this scope; not general calibration certification",
        "interval_adjustment": "none",
    }


def measurement_result(score, variance, scope, support, calibration=None):
    """Cold serializable result preserving support and matched calibration status."""
    if not scope or support not in ("observed", "model_borrowed", "unobserved"):
        raise ValueError("explicit scope and support status required")
    if calibration is not None and calibration["scope"] != scope:
        raise ValueError("calibration scope does not match score scope")
    if support == "unobserved":
        if score is not None or variance is not None:
            raise ValueError("unobserved without model must be null, not zero")
        lower = upper = None
    else:
        if (
            score is None
            or variance is None
            or not np.isfinite(score)
            or not np.isfinite(variance)
            or variance < 0
        ):
            raise ValueError("finite score and nonnegative model variance required")
        half = 1.96 * np.sqrt(variance)
        lower, upper = float(score - half), float(score + half)
    return {
        "schema_version": "grain_uncertainty_contract_v1",
        "scope": dict(scope),
        "score": score,
        "variance": variance,
        "lower": lower,
        "upper": upper,
        "support": support,
        "interval_kind": "gaussian_working_moment_interval"
        if score is not None
        else None,
        "calibration_check": calibration["decision"]
        if calibration is not None
        else "NOT_EVALUATED",
        "calibration_scope": calibration["scope"] if calibration is not None else None,
        "human_validity": "NOT_ESTABLISHED_BY_THIS_CONTRACT",
    }

#!/usr/bin/env python3
"""Explicit affine chart averaging versus reference-mean score anchoring."""

from __future__ import annotations

import numpy as np


def combine_affine_references(charts, weights=None):
    """Return both anchor conventions without changing the common linear operator.

    Z/h are reference P/Q states; G/g are coefficients and b is the local anchor.
    Means of centered affine parameters are not means of whole affine functions.
    """
    if not charts:
        raise ValueError("At least one local affine chart required")
    arrays = {
        k: np.stack([np.asarray(r[k], float) for r in charts])
        for k in ("Z", "h", "G", "g", "b")
    }
    n = len(charts)
    if (
        arrays["Z"].ndim != 3
        or arrays["Z"].shape != arrays["G"].shape
        or arrays["h"].shape != (n, arrays["Z"].shape[2])
        or arrays["g"].shape != arrays["h"].shape
        or arrays["b"].shape != (n,)
        or min(arrays["Z"].shape) < 1
        or not all(np.isfinite(v).all() for v in arrays.values())
    ):
        raise ValueError(
            "Finite aligned P/Q references, operators and scalar anchors required"
        )
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(
            "Finite nonnegative reference weights with positive total required"
        )
    w = w / w.sum()
    mean = {k: np.einsum("r,r...->...", w, values) for k, values in arrays.items()}
    p_cross = float(
        np.dot(w, np.sum(arrays["G"] * arrays["Z"], axis=(1, 2)))
        - np.sum(mean["G"] * mean["Z"])
    )
    q_cross = float(
        np.dot(w, np.sum(arrays["g"] * arrays["h"], axis=1))
        - np.dot(mean["g"], mean["h"])
    )
    return {
        **mean,
        "weights": w,
        "p_reference_operator_covariance": p_cross,
        "q_reference_operator_covariance": q_cross,
        "mean_anchor_minus_chart_mean": p_cross + q_cross,
        "b_mean_anchor": float(mean["b"]),
        "b_chart_mean": float(mean["b"] - p_cross - q_cross),
    }


def score_affine_reference(reference, p, q, convention="mean_anchor"):
    """Score same-coordinate states using a named convention; never clip the result."""
    if convention not in ("mean_anchor", "chart_mean"):
        raise ValueError("Anchor convention must be explicit and recognized")
    p, q = np.asarray(p, float), np.asarray(q, float)
    if (
        p.shape != reference["Z"].shape
        or q.shape != reference["h"].shape
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Query state shape/value mismatch")
    return float(
        reference[f"b_{convention}"]
        + np.sum(reference["G"] * (p - reference["Z"]))
        + np.dot(reference["g"], q - reference["h"])
    )

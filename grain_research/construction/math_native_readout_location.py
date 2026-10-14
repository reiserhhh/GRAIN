#!/usr/bin/env python3
"""Symmetric categorical interaction account, not a new prediction model."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction.math_joint_reading import (
    BASIS,
    coefficients,
    reconstruct,
)
from grain_research.construction.math_native_joint import center_logits, probability

VALUES = np.arange(10) / 9
PARTS = ("native", "additive_readout", "joint_field_lift")
UNITS = ("score", "log_odds")


def readout(logits):
    """Read two fixed native units from a finite ten-digit natural coordinate."""
    z = center_logits(logits)
    if z.shape[-1] != 10:
        raise ValueError("The frozen ten-digit response alphabet is required")
    return np.stack(
        (
            probability(z) @ VALUES,
            logsumexp(z[..., 5:], -1) - logsumexp(z[..., :5], -1),
        ),
        axis=-1,
    )


def derivatives(logits):
    """Return exact gradients/Hessians of expectation and grouped log odds.

    The expectation Hessian is not a Fisher matrix and need not be positive
    semidefinite. Grouped log odds subtracts two covariance Hessians.
    """
    z = center_logits(logits)
    if z.shape[-1] != 10:
        raise ValueError("Ten finite natural coordinates required")
    p = probability(z)
    g = p * (VALUES - (p @ VALUES)[..., None])
    expectation_h = (
        np.eye(10) * g[..., :, None]
        - p[..., :, None] * g[..., None, :]
        - g[..., :, None] * p[..., None, :]
    )
    high, low = np.zeros_like(z), np.zeros_like(z)
    high[..., 5:] = probability(z[..., 5:])
    low[..., :5] = probability(z[..., :5])
    grouped_h = (
        np.eye(10) * (high - low)[..., :, None]
        - high[..., :, None] * high[..., None, :]
        + low[..., :, None] * low[..., None, :]
    )
    return np.stack((g, high - low), axis=-2), np.stack(
        (expectation_h, grouped_h), axis=-3
    )


def decompose(logits):
    """Preserve all four characters and split exact readout from joint-field lift."""
    z = center_logits(logits)
    if z.shape != (4, 10):
        raise ValueError("Four ordered complete ten-digit corners required")
    natural = coefficients(z)
    additive = z - BASIS[3, :, None] * natural[3]
    if not np.allclose(coefficients(additive)[3], 0, rtol=0, atol=1e-12):
        raise ValueError("Declared additive field retains a joint natural component")
    native, main = readout(z), readout(additive)
    corners = {
        "native": native,
        "additive_readout": main,
        "joint_field_lift": native - main,
    }
    chars = {k: coefficients(v) for k, v in corners.items()}
    error = max(
        float(
            np.max(
                abs(
                    chars["native"]
                    - chars["additive_readout"]
                    - chars["joint_field_lift"]
                )
            )
        ),
        float(np.max(abs(reconstruct(natural) - z))),
        *(float(np.max(abs(reconstruct(v) - corners[k]))) for k, v in chars.items()),
    )
    return {
        "z": z,
        "natural": natural,
        "additive": additive,
        "corners": corners,
        "characters": chars,
        "error": error,
    }


def integral_account(logits, order=32):
    """Check the fixed mixed-Hessian rectangle and all joint-lift line integrals."""
    if not isinstance(order, int) or order < 2:
        raise ValueError("A fixed positive quadrature order is required")
    result = decompose(logits)
    z0, x, y, _ = result["natural"]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    rectangle = z0 + nodes[:, None, None] * x + nodes[None, :, None] * y
    _, h = derivatives(rectangle)
    mixed = np.einsum("i,abqij,j->abq", x, h, y)
    rectangle_integral = np.einsum("a,b,abq->q", weights, weights, mixed) / 4
    delta = result["z"] - result["additive"]
    lines = (
        result["additive"][:, None] + (nodes[None, :, None] + 1) * delta[:, None] / 2
    )
    grad, _ = derivatives(lines)
    corner_lifts = np.einsum("t,ctqi,ci->cq", weights / 2, grad, delta)
    lift_integral = coefficients(corner_lifts)
    error = max(
        float(
            np.max(
                abs(rectangle_integral - result["characters"]["additive_readout"][3])
            )
        ),
        float(np.max(abs(lift_integral - result["characters"]["joint_field_lift"]))),
    )
    return {"rectangle": rectangle_integral, "lift": lift_integral, "error": error}


def author_account(decomposition, law):
    """Compose a supplied positive evidence law, not a human personality score."""
    p = np.asarray(law, float)
    if (
        p.shape != (4,)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("A positive normalized four-corner evidence law is required")
    moments = BASIS @ p
    scores = {k: p @ decomposition["corners"][k] for k in PARTS}
    error = max(
        *(
            float(np.max(abs(scores[k] - moments @ decomposition["characters"][k])))
            for k in PARTS
        ),
        float(
            np.max(
                abs(
                    scores["native"]
                    - scores["additive_readout"]
                    - scores["joint_field_lift"]
                )
            )
        ),
    )
    return scores, error


def signed_account(total, additive, lift):
    """Keep both signed components and their cross term; do not add RMS shares."""
    t, a, b = (np.asarray(x, float).ravel() for x in (total, additive, lift))
    if (
        t.shape != a.shape
        or a.shape != b.shape
        or not len(t)
        or not np.isfinite(t + a + b).all()
    ):
        raise ValueError("Matched nonempty finite signed observations required")
    norm = float(t @ t)
    return {
        "n": len(t),
        "native_rms": float(np.sqrt(np.mean(t * t))),
        "additive_readout_rms": float(np.sqrt(np.mean(a * a))),
        "joint_field_lift_rms": float(np.sqrt(np.mean(b * b))),
        "cross_second_moment": float(2 * np.mean(a * b)),
        "additive_signed_alignment": float(a @ t / norm) if norm else None,
        "lift_signed_alignment": float(b @ t / norm) if norm else None,
        "opposite_sign_count": int(np.sum(a * b < 0)),
        "identity_error": float(np.max(abs(t - a - b))),
        "second_moment_error": float(
            abs(np.mean(t * t) - np.mean(a * a) - np.mean(b * b) - 2 * np.mean(a * b))
        ),
    }

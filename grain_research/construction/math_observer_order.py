#!/usr/bin/env python3
"""Finite observer-order diagnostics, not personality scoring or calibration."""

from __future__ import annotations

import itertools

import numpy as np

JOINT_SIGN = np.array([1.0, -1.0, -1.0, 1.0])
JOINT_PLUS = (1 + 0.5 * JOINT_SIGN) / 4
JOINT_MINUS = (1 - 0.5 * JOINT_SIGN) / 4


def checked_pair(first, second, tolerance=1e-10):
    """Validate nonempty finite matched observation units and absolute tie scale."""
    f, n = np.asarray(first, float), np.asarray(second, float)
    if (
        f.ndim != 1
        or n.shape != f.shape
        or f.size < 2
        or not np.isfinite(f).all()
        or not np.isfinite(n).all()
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Matched finite vectors and nonnegative tolerance required")
    return f, n


def contrast_class(first, second, tolerance=1e-10):
    """Keep reversed order and ties separate without orienting from the outcome."""
    if not np.isfinite(first) or not np.isfinite(second):
        raise ValueError("Two finite contrasts required")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Nonnegative finite tolerance required")
    a, b = abs(first) <= tolerance, abs(second) <= tolerance
    if a and b:
        return "both_tied"
    if a or b:
        return "one_sided_tie"
    return "concordant" if np.sign(first) == np.sign(second) else "reversed"


def pair_orders(first, second, tolerance=1e-10):
    """Enumerate every unordered event pair with independently declared units."""
    f, n = checked_pair(first, second, tolerance)
    return [
        {
            "left": i,
            "right": j,
            "first_contrast": float(f[i] - f[j]),
            "second_contrast": float(n[i] - n[j]),
            "classification": contrast_class(f[i] - f[j], n[i] - n[j], tolerance),
        }
        for i, j in itertools.combinations(range(len(f)), 2)
    ]


def affine_certificate(first, second, tolerance=1e-10):
    """Describe affinity on this support; never return a calibrated predictor.

    Positive affinity is required for identical weak orders under all interior
    probability mixtures. A negative slope can have zero residual but reverses
    those orders. Undefined relative residuals stay None in constant units.
    """
    f, n = checked_pair(first, second, tolerance)
    fc, nc = f - f.mean(), n - n.mean()
    f_constant = bool(np.max(abs(fc)) <= tolerance)
    n_constant = bool(np.max(abs(nc)) <= tolerance)
    if f_constant:
        return {
            "first_constant": True,
            "second_constant": n_constant,
            "slope": None,
            "offset": None,
            "relative_residual": None,
            "max_absolute_residual": float(np.max(abs(nc))),
            "positive_affine_on_support": n_constant,
        }
    slope = float(fc @ nc / (fc @ fc))
    residual = nc - slope * fc
    maximum = float(np.max(abs(residual)))
    return {
        "first_constant": False,
        "second_constant": n_constant,
        "slope": slope,
        "offset": float(n.mean() - slope * f.mean()),
        "relative_residual": float(np.linalg.norm(residual) / np.linalg.norm(nc))
        if not n_constant
        else None,
        "max_absolute_residual": maximum,
        "positive_affine_on_support": bool(slope > 0 and maximum <= tolerance),
    }


def evidence_mean(values, law):
    """Integrate a numeric observation under a supplied normalized evidence law."""
    f, p = np.asarray(values, float), np.asarray(law, float)
    if (
        f.ndim != 1
        or p.shape != f.shape
        or not len(f)
        or not np.isfinite(f).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Finite observation and positive normalized law required")
    return float(p @ f)


def fixed_margin_contrast(values):
    """Keep content and ownership marginals equal while changing their coupling."""
    f = np.asarray(values, float)
    if f.shape != (4,) or not np.isfinite(f).all():
        raise ValueError("Four complete finite corners required")
    return evidence_mean(f, JOINT_PLUS) - evidence_mean(f, JOINT_MINUS)

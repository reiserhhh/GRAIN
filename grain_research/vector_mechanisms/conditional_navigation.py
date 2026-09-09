#!/usr/bin/env python3
"""Finite response diagnostics; no human labels or neural model dependencies."""

from __future__ import annotations

import numpy as np


def finite_pair(actual, predicted):
    """Require identically shaped, nonempty finite scalar response observations."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.ndim != 1 or a.shape != p.shape or not len(a):
        raise ValueError("Matched nonempty response vectors required")
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite response")
    return a, p


def response_metrics(actual, predicted):
    """Separate signed association, gain, amplitude and error against zero response."""
    a, p = finite_pair(actual, predicted)
    energy, amplitude = float(a @ a), float(np.mean(abs(a)))
    return {
        "n": len(a),
        "r": float(np.corrcoef(a, p)[0, 1])
        if len(a) >= 3 and min(np.ptp(a), np.ptp(p)) > 1e-12
        else None,
        "slope_pred_on_actual": float(a @ p / energy) if energy > 1e-24 else None,
        "amplitude_ratio": float(np.mean(abs(p)) / amplitude)
        if amplitude > 1e-12
        else None,
        "mean_abs_actual": amplitude,
        "mean_abs_predicted": float(np.mean(abs(p))),
        "mse": float(np.mean((a - p) ** 2)),
        "mse_over_zero": float(np.sum((a - p) ** 2) / energy)
        if energy > 1e-24
        else None,
        "sign_agreement": float(np.mean(np.sign(a) == np.sign(p))),
    }


def error_accounting(native, fixed, program):
    """Account for squared error without treating correlated pieces as orthogonal."""
    native, fixed = finite_pair(native, fixed)
    _, program = finite_pair(native, program)
    a, b = native - fixed, fixed - program
    total = float(np.mean((native - program) ** 2))
    first, second = float(np.mean(a**2)), float(np.mean(b**2))
    cross = float(2 * np.mean(a * b))
    return {
        "native_minus_program_mse": total,
        "native_minus_fixed_mse": first,
        "fixed_minus_program_mse": second,
        "twice_error_cross": cross,
        "accounting_error": abs(total - first - second - cross),
    }


def projected_response_error(early_delta, upper_delta, predicted_delta, basis, reader):
    """Decompose a response with a frozen residual span; all center terms cancel.

    Later-layer differences make this a diagnostic oracle, not an input to an
    independent predictor. Orthogonal state terms can correlate after readout.
    """
    x, y = finite_pair(early_delta, upper_delta)
    _, p = finite_pair(y, predicted_delta)
    v, l = np.asarray(basis, float), np.asarray(reader, float)
    if v.ndim != 2 or v.shape[1] != len(x) or l.shape != x.shape:
        raise ValueError("Basis and reader coordinates must match the state")
    if not np.isfinite(v).all() or not np.isfinite(l).all():
        raise ValueError("Nonfinite basis or reader")
    if not np.allclose(v @ v.T, np.eye(len(v)), atol=1e-6):
        raise ValueError("Frozen basis must be orthonormal")
    residual = y - x
    outside = residual - (residual @ v.T) @ v
    inside = y - p - outside
    total = y - p
    a, b = float(outside @ l), float(inside @ l)
    t = float(total @ l)
    return {
        "state_outside_sq": float(outside @ outside),
        "state_inside_sq": float(inside @ inside),
        "state_cross": float(2 * outside @ inside),
        "state_total_sq": float(total @ total),
        "reader_outside": a,
        "reader_inside": b,
        "reader_outside_sq": a**2,
        "reader_inside_sq": b**2,
        "reader_cross": 2 * a * b,
        "reader_total_sq": t**2,
        "reader_accounting_error": abs(t**2 - a**2 - b**2 - 2 * a * b),
    }


def scalar_gain_oracle(actual, predicted):
    """Fit one retrospective gain; never export or promote this as a predictor."""
    a, p = finite_pair(actual, predicted)
    denom = float(p @ p)
    gamma = float(p @ a / denom) if denom > 1e-24 else 0.0
    raw, fitted = float(np.sum((a - p) ** 2)), float(np.sum((a - gamma * p) ** 2))
    return {
        "oracle_gain": gamma,
        "raw_sse": raw,
        "oracle_sse": fitted,
        "oracle_relative_sse_reduction": (raw - fitted) / raw if raw > 1e-24 else None,
        "inference_usable": False,
    }

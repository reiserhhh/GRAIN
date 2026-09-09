#!/usr/bin/env python3
"""Shared source-fitted operators and explicit finite-change accounting."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def finite_panel(x, y):
    """Require matched author-by-view factor and scalar-response panels."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        x.ndim != 3
        or y.shape != x.shape[:2]
        or min(x.shape) < 1
        or len(x) < 3
        or x.shape[1] < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Matched finite author/view panels required")
    return x, y


def estimand_rows(x, y, kind):
    """Keep between, within and transition estimands distinct."""
    x, y = finite_panel(x, y)
    if kind == "between":
        return x.mean(1), y.mean(1)
    if kind == "within":
        return (
            (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[2]),
            (y - y.mean(1, keepdims=True)).ravel(),
        )
    if kind == "transition":
        dx, dy = np.diff(x, axis=1), np.diff(y, axis=1)
        return (
            (dx - dx.mean(0, keepdims=True)).reshape(-1, x.shape[2]),
            (dy - dy.mean(0, keepdims=True)).ravel(),
        )
    raise ValueError("Unknown estimation level")


def fit_operator(x, y, kind, alpha=100.0):
    """Fit once on reference authors; return raw-coordinate slopes and intercept."""
    a, b = estimand_rows(x, y, kind)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Nonnegative finite penalty required")
    center = a.mean(0)
    scale = a.std(0)
    active = scale > 1e-12
    scale = np.where(active, scale, 1.0)
    model = Ridge(alpha=alpha, fit_intercept=(kind == "between"))
    model.fit((a - center) / scale * active, b)
    coef = model.coef_ / scale * active
    intercept = float(model.intercept_ - center @ coef)
    result = {
        "kind": kind,
        "alpha": float(alpha),
        "authors": len(x),
        "rows": len(a),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
    }
    if kind == "transition":
        result["transition_x_center"] = np.diff(x, axis=1).mean(0).tolist()
        result["transition_y_center"] = np.diff(y, axis=1).mean(0).tolist()
    return result


def correlation(x, y):
    """Return no correlation rather than zero for an undefined constant series."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite([x, y]).all():
        raise ValueError("Matched finite vectors required")
    a, b = x - x.mean(), y - y.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / norm, -1, 1)) if len(x) >= 3 and norm > 1e-14 else None


def paired_summary(actual, predicted, seed=20260910, draws=1999, calibrated=True):
    """Bootstrap complete author pairs; expose scale as well as association."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    r = correlation(actual, predicted)
    rng, values = np.random.default_rng(seed), []
    for _ in range(draws):
        idx = rng.integers(0, len(actual), len(actual))
        value = correlation(actual[idx], predicted[idx])
        if value is not None:
            values.append(value)
    result = {
        "n": len(actual),
        "r": r,
        "r_lower": float(np.quantile(values, 0.025)) if values else None,
        "r_upper": float(np.quantile(values, 0.975)) if values else None,
        "valid_bootstraps": len(values),
    }
    if calibrated:
        energy = float(np.mean(actual**2))
        result.update(
            actual_rms=float(np.sqrt(energy)),
            predicted_rms=float(np.sqrt(np.mean(predicted**2))),
            mse_over_zero=float(np.mean((actual - predicted) ** 2) / energy)
            if energy > 1e-20
            else None,
            sign_match=float(np.mean(np.sign(actual) == np.sign(predicted))),
        )
    return result


def response_channels(native, functional, represented):
    """Split a finite native change without treating remainder as pure noise."""
    native, functional, represented = (
        np.asarray(v, float) for v in (native, functional, represented)
    )
    if (
        native.ndim != 1
        or native.shape != functional.shape
        or native.shape != represented.shape
    ):
        raise ValueError("Equal finite response vectors required")
    if not np.isfinite([native, functional, represented]).all():
        raise ValueError("Equal finite response vectors required")
    channels = np.column_stack(
        (represented, functional - represented, native - functional)
    )
    gram = channels.T @ channels / len(channels)
    return channels, {
        "channel_order": ["grain", "query_remainder", "reader_gap"],
        "second_moment": gram.tolist(),
        "native_energy": float(np.mean(native**2)),
        "sum_error": float(np.max(np.abs(channels.sum(1) - native))),
        "energy_error": float(abs(gram.sum() - np.mean(native**2))),
    }


def removal_coordinate(values, indices):
    """Separate removed fraction and signed contrast in an excerpt-mean change."""
    values = np.asarray(values, float)
    indices = np.asarray(indices)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or not 0 < len(indices) < len(values)
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(values))
    ):
        raise ValueError(
            "Unique valid removal indices and retained observations required"
        )
    keep = np.ones(len(values), bool)
    keep[indices] = False
    full, retained, removed = values.mean(), values[keep].mean(), values[~keep].mean()
    fraction, contrast = len(indices) / len(values), retained - removed
    return {
        "full": float(full),
        "retained": float(retained),
        "removed": float(removed),
        "fraction": float(fraction),
        "contrast": float(contrast),
        "delta": float(retained - full),
        "identity_error": float(abs(retained - full - fraction * contrast)),
    }


def signed_energy_allocation(native, channels):
    """Retain signed cancellation rather than label squared pieces variance shares."""
    native, channels = np.asarray(native, float), np.asarray(channels, float)
    if (
        native.ndim != 1
        or channels.ndim != 2
        or len(channels) != len(native)
        or not np.isfinite(native).all()
        or not np.isfinite(channels).all()
    ):
        raise ValueError("Finite native responses and matching channel rows required")
    energy = float(np.mean(native**2))
    if energy <= 1e-20:
        raise ValueError("Native response energy is zero")
    return np.mean(channels * native[:, None], axis=0) / energy

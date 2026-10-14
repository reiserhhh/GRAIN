#!/usr/bin/env python3
"""Finite coupling coordinates and sharp reading bounds, not fitted traits."""

from __future__ import annotations

import numpy as np

CYCLE = np.array([[1.0, -1.0], [-1.0, 1.0]])


def coordinates(tables):
    """Decompose nonnegative 2x2 tables into their margins and one cycle.

    The coordinate is in count/mass units. Missing groups have zero mass and
    zero feasible interval; this does not imply a measured conditional score.
    """
    x = np.asarray(tables, dtype=float)
    if x.ndim < 2 or x.shape[-2:] != (2, 2):
        raise ValueError("A complete two-by-two coupling support is required")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Coupling masses must be finite and nonnegative")
    rows, columns = x.sum(axis=-1), x.sum(axis=-2)
    total = rows.sum(axis=-1)
    independent = np.zeros_like(x)
    np.divide(
        rows[..., :, None] * columns[..., None, :],
        total[..., None, None],
        out=independent,
        where=total[..., None, None] > 0,
    )
    coordinate = x[..., 0, 0] - independent[..., 0, 0]
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return {
        "rows": rows,
        "columns": columns,
        "total": total,
        "observed": total > 0,
        "independent": independent,
        "coordinate": coordinate,
        "lower_coordinate": lower - independent[..., 0, 0],
        "upper_coordinate": upper - independent[..., 0, 0],
    }


def reconstruct(rows, columns, coordinate):
    """Recover a feasible coupling from equal-total margins and its cycle.

    Real masses are accepted: independent reference tables need not contain
    integer counts. Invalid coordinates are rejected, not clipped to fit data.
    """
    rows, columns, value = (
        np.asarray(rows, float),
        np.asarray(columns, float),
        np.asarray(coordinate, float),
    )
    if (
        rows.ndim < 1
        or rows.shape[-1] != 2
        or columns.shape != rows.shape
        or value.shape != rows.shape[:-1]
        or not np.isfinite(rows).all()
        or not np.isfinite(columns).all()
        or not np.isfinite(value).all()
        or np.any(rows < 0)
        or np.any(columns < 0)
    ):
        raise ValueError("Legal margins and aligned finite coordinates required")
    total = rows.sum(axis=-1)
    if not np.allclose(total, columns.sum(axis=-1), rtol=1e-12, atol=1e-12):
        raise ValueError("Row and column margins have different total mass")
    base = np.zeros(rows.shape[:-1] + (2, 2))
    np.divide(
        rows[..., :, None] * columns[..., None, :],
        total[..., None, None],
        out=base,
        where=total[..., None, None] > 0,
    )
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total) - base[..., 0, 0]
    upper = np.minimum(rows[..., 0], columns[..., 0]) - base[..., 0, 0]
    tolerance = 1e-12 * np.maximum(1.0, total)
    if np.any(value < lower - tolerance) or np.any(value > upper + tolerance):
        raise ValueError("Cycle lies outside the fixed-margin feasible interval")
    return base + value[..., None, None] * CYCLE


def response_account(tables, kernels):
    """Map target-independent pairing coordinates to all registered readings.

    Tables are group/behavior/focus; kernels are group/behavior/focus followed
    by any output axes. Bounds are sharp per output, but their witnesses need
    not be one common policy for every output. They are not confidence bounds.
    """
    x, k = np.asarray(tables, float), np.asarray(kernels, float)
    if (
        x.ndim != 3
        or x.shape[1:] != (2, 2)
        or k.ndim < 3
        or k.shape[:3] != x.shape
        or not np.isfinite(k).all()
    ):
        raise ValueError("Aligned complete group/behavior/focus kernels required")
    c = coordinates(x)
    total = float(c["total"].sum())
    if total <= 0:
        raise ValueError("An unobserved author cannot receive a zero reading")
    output_axes = (1,) * (k.ndim - 3)

    def expand(t):
        return t.reshape(t.shape + output_axes)

    def integrate(t):
        return (expand(t) * k).sum(axis=(0, 1, 2)) / total

    gamma = k[:, 0, 0] - k[:, 0, 1] - k[:, 1, 0] + k[:, 1, 1]
    cycles = c["coordinate"].reshape((-1,) + output_axes) * gamma / total
    independent = integrate(c["independent"])
    reference = integrate(np.repeat(c["rows"][..., None] / 2, 2, axis=-1))
    lower = c["lower_coordinate"].reshape((-1,) + output_axes)
    upper = c["upper_coordinate"].reshape((-1,) + output_axes)
    lower_witness = np.where(gamma >= 0, lower, upper)
    upper_witness = np.where(gamma >= 0, upper, lower)
    return {
        "reference": reference,
        "independent": independent,
        "marginal": independent - reference,
        "cycle": cycles.sum(axis=0),
        "cycle_by_group": cycles,
        "observed": integrate(x),
        "reconstructed": independent + cycles.sum(axis=0),
        "lower": independent + (lower_witness * gamma).sum(axis=0) / total,
        "upper": independent + (upper_witness * gamma).sum(axis=0) / total,
        "lower_witness": lower_witness,
        "upper_witness": upper_witness,
        "gamma": gamma,
        "coverage": c["observed"],
    }

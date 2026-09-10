#!/usr/bin/env python3
"""Separate local marginal spread, signed alignment and reference interaction."""

from __future__ import annotations

import numpy as np


def cross_covariance(x, y):
    """Return population-divisor cross-covariance on matched local observations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Two aligned observation matrices are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite observations required")
    return (x - x.mean(0)).T @ (y - y.mean(0)) / len(x)


def local_geometry(grain, functional):
    """Factor covariance and compare one predetermined cyclic pairing shift.

    Zero-spread rho is set to zero for exact accounting and marked degenerate;
    it must not be reported as an observed zero Pearson correlation.
    """
    g, p = np.asarray(grain, float), np.asarray(functional, float)
    if g.ndim != 2 or p.shape != (len(g),) or len(g) < 2:
        raise ValueError("At least two aligned comments required")
    if not np.isfinite(g).all() or not np.isfinite(p).all():
        raise ValueError("Finite observations required")
    gc, pc = g - g.mean(0), p - p.mean()
    sg, sp = np.sqrt(np.mean(gc**2, 0)), float(np.sqrt(np.mean(pc**2)))
    amplitude = sg * sp
    signed = cross_covariance(g, p[:, None])[:, 0]
    shifted = cross_covariance(g, np.roll(p, 1)[:, None])[:, 0]
    rho = np.divide(signed, amplitude, out=np.zeros_like(signed), where=amplitude > 0)
    shifted_rho = np.divide(
        shifted, amplitude, out=np.zeros_like(shifted), where=amplitude > 0
    )
    return {
        "sg": sg,
        "sp": np.full_like(sg, sp),
        "A": amplitude,
        "rho": rho,
        "S": signed,
        "shifted_S": shifted,
        "shifted_rho": shifted_rho,
        "matched_excess": signed - shifted,
        "degenerate": amplitude == 0,
        "identity_error": float(np.max(abs(signed - amplitude * rho))),
    }


def product_components(amplitude, alignment, a0, r0):
    """Expand a product around declared reference means without dropping interaction."""
    a, r = np.asarray(amplitude, float), np.asarray(alignment, float)
    if a.shape != r.shape or not np.isfinite([a, r]).all():
        raise ValueError("Finite, aligned arrays required")
    if not np.isfinite([a0, r0]).all():
        raise ValueError("Finite reference means required")
    return {
        "reference": np.full_like(a, a0 * r0),
        "spread_component": r0 * (a - a0),
        "alignment_component": a0 * (r - r0),
        "interaction_component": (a - a0) * (r - r0),
    }

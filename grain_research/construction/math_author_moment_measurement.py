#!/usr/bin/env python3
"""Finite distribution-to-observation functionals and sharp moment sets."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog


def cube(dimension=6):
    """Return lexicographic +/-1 configurations, subset masks and characters."""
    if not isinstance(dimension, int) or dimension < 1 or dimension > 10:
        raise ValueError("positive bounded integer dimension required")
    masks = np.asarray(list(itertools.product((0, 1), repeat=dimension)), int)
    points = 2 * masks - 1
    characters = np.prod(points[:, None, :] ** masks[None, :, :], axis=-1)
    return points, masks, characters.astype(float)


def probability(weights, dimension=6):
    """Validate, rather than repair or silently renormalize, a finite law."""
    p = np.asarray(weights, float)
    if (
        p.shape != (2**dimension,)
        or not np.isfinite(p).all()
        or np.min(p) < -1e-10
        or abs(p.sum() - 1) > 1e-9
    ):
        raise ValueError("finite nonnegative unit-mass law required")
    return p


def moment_design(order, dimension=6):
    """Rows are all observed moments through order, including unit mass."""
    _, masks, w = cube(dimension)
    if not isinstance(order, int) or not 0 <= order <= dimension:
        raise ValueError("moment order outside the finite cube")
    return w[:, masks.sum(1) <= order].T


def moments(weights, dimension=6):
    """Return every subset moment; no independence assumption is imposed."""
    _, _, w = cube(dimension)
    return w.T @ probability(weights, dimension)


def parity_profiles(subset, strength=0.5, dimension=6):
    """Full-support pair with equal lower moments and one changed joint moment."""
    _, masks, w = cube(dimension)
    mask = np.asarray(subset)
    found = (
        np.flatnonzero(np.all(masks == mask, axis=1))
        if mask.shape == (dimension,)
        else []
    )
    if len(found) != 1 or not mask.sum() or not 0 < strength < 1:
        raise ValueError(
            "nonempty binary subset and strictly interior strength required"
        )
    chi = w[:, found[0]]
    return (1 + strength * chi) / len(w), (1 - strength * chi) / len(w)


def integrate(values, weights, dimension=6):
    """Average per-configuration readings, not an LLM whole-history judgment."""
    v = np.asarray(values, float)
    if v.ndim < 1 or v.shape[0] != 2**dimension or not np.isfinite(v).all():
        raise ValueError("complete finite observation table required")
    return np.tensordot(probability(weights, dimension), v, axes=(0, 0))


def coefficients(values, dimension=6):
    """Uniform finite Fourier coefficients in the declared lexicographic basis."""
    v = np.asarray(values, float)
    if v.ndim < 1 or v.shape[0] != 2**dimension or not np.isfinite(v).all():
        raise ValueError("complete finite observation table required")
    _, _, w = cube(dimension)
    return np.tensordot(w.T / len(w), v, axes=(1, 0))


def sharp_interval(values, observed_moments, order, dimension=6):
    """Attain finite min/max readings with primal distributions and dual bounds.

    The interval is an identification set for fixed exact moments, not a
    sampling CI. No reconstruction penalty, tuning parameter or prior is used.
    """
    v = np.asarray(values, float)
    a = moment_design(order, dimension)
    mu = np.asarray(observed_moments, float)
    if v.shape != (2**dimension,) or mu.shape != (len(a),):
        raise ValueError("response or moment shape mismatch")
    if not np.isfinite(v).all() or not np.isfinite(mu).all() or abs(mu[0] - 1) > 1e-9:
        raise ValueError("finite unit-mass moments required")
    options = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
    result = {}
    for side, sign in (("lower", 1), ("upper", -1)):
        fit = linprog(
            sign * v, A_eq=a, b_eq=mu, bounds=(0, None), method="highs", options=options
        )
        if not fit.success:
            raise ValueError(f"moment set LP {side} failed:{fit.status}:{fit.message}")
        p, lam = fit.x, sign * fit.eqlin.marginals
        bound = float(v @ p)
        slack = v - a.T @ lam if side == "lower" else a.T @ lam - v
        error = max(
            float(np.max(np.abs(a @ p - mu))),
            max(0.0, float(-p.min())),
            max(0.0, float(-slack.min())),
            abs(bound - float(mu @ lam)),
        )
        if error > 1e-8:
            raise ArithmeticError("unverified primal/dual moment certificate")
        result[side] = {"value": bound, "distribution": p, "dual": lam, "error": error}
    result["width"] = result["upper"]["value"] - result["lower"]["value"]
    if result["width"] < -1e-8:
        raise ArithmeticError("reversed identification interval")
    return result

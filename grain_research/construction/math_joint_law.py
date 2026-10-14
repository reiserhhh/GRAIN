#!/usr/bin/env python3
"""Finite Boolean response laws under a declared, complete factorial design.

Möbius coefficients use an explicit all-zero anchor. Fourier coefficients
use a declared independent product reference, not a natural-text population.
Neither is automatically a psychological response parameter.
"""

from __future__ import annotations

import numpy as np


def _table(values):
    """Validate an aligned mask-indexed scalar or vector response table."""
    f = np.asarray(values, dtype=float)
    n = f.shape[0] if f.ndim else 0
    if n < 2 or n & (n - 1) or not np.isfinite(f).all():
        raise ValueError("finite, complete power-of-two response table required")
    return f.copy(), n.bit_length() - 1


def mobius(values):
    """Return anchored interaction coefficients, including the intercept.

    Row m is the cell whose present coordinates are the set bits of m.
    Trailing axes are independent output coordinates, not crossed factors.
    """
    out, d = _table(values)
    for j in range(d):
        for m in range(len(out)):
            if m & (1 << j):
                out[m] -= out[m ^ (1 << j)]
    return out


def zeta(coefficients):
    """Invert the Boolean Möbius transform without fitting missing cells."""
    out, d = _table(coefficients)
    for j in range(d):
        for m in range(len(out)):
            if m & (1 << j):
                out[m] += out[m ^ (1 << j)]
    return out


def product_fourier(values, probabilities):
    """Decompose responses under explicitly independent Bernoulli factors.

    The same observed cell table may have different component energies under
    a different reference law. No coordinates or conditions are discarded.
    """
    f, d = _table(values)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (d,) or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("one strictly interior probability per crossed factor")
    bits = (np.arange(len(f))[:, None] >> np.arange(d)) & 1
    weights = np.prod(np.where(bits, p, 1 - p), axis=1)
    standardized = (bits - p) / np.sqrt(p * (1 - p))
    basis = np.ones((len(f), len(f)))
    for m in range(1, len(f)):
        included = [j for j in range(d) if m & (1 << j)]
        basis[:, m] = np.prod(standardized[:, included], axis=1)
    flat = f.reshape(len(f), -1)
    coefficients = basis.T @ (weights[:, None] * flat)
    reconstruction = basis @ coefficients
    energies = np.sum(coefficients**2, axis=1)
    mean = weights @ flat
    variance = float(np.sum(weights[:, None] * (flat - mean) ** 2))
    return {
        "coefficients": coefficients.reshape(f.shape),
        "weights": weights,
        "basis": basis,
        "energies": energies,
        "variance": variance,
        "reconstruction_error": float(np.max(np.abs(reconstruction - flat))),
        "gram_error": float(
            np.max(np.abs(basis.T @ (weights[:, None] * basis) - np.eye(len(f))))
        ),
        "variance_closure_error": abs(float(energies[1:].sum()) - variance),
    }


def design_matrix(masks, dimension):
    """Build all subset-monomial columns without assuming missing terms zero."""
    masks = np.asarray(masks)
    if dimension < 1 or masks.ndim != 1 or not np.issubdtype(masks.dtype, np.integer):
        raise ValueError("integer cell addresses and positive dimension required")
    if np.any((masks < 0) | (masks >= 1 << dimension)):
        raise ValueError("cell address outside declared Boolean lattice")
    subsets = np.arange(1 << dimension)
    return ((masks[:, None] & subsets) == subsets).astype(float)


def support_witness():
    """Three identical singleton sources permit three different joint laws."""
    x = design_matrix(np.array([0, 1, 2]), 2)
    worlds = {
        "additive": np.array([0.2, 0.15, 0.1, 0.0]),
        "synergy": np.array([0.2, 0.15, 0.1, 0.4]),
        "cancellation": np.array([0.2, 0.15, 0.1, -0.4]),
    }
    full = design_matrix(np.arange(4), 2)
    null = np.array([0.0, 0.0, 0.0, 1.0])
    return {
        "source_cells": [0, 1, 2],
        "singleton_rank": int(np.linalg.matrix_rank(x)),
        "duplicate_singleton_rank": int(np.linalg.matrix_rank(np.tile(x, (10, 1)))),
        "full_joint_rank": int(np.linalg.matrix_rank(full)),
        "null_vector": null.tolist(),
        "source_null_error": float(np.max(np.abs(x @ null))),
        "joint_null_action": float((full @ null)[3]),
        "worlds": {
            name: {"coefficients": b.tolist(), "cells": (full @ b).tolist()}
            for name, b in worlds.items()
        },
        "scope": "exact design identifiability witness, not observed human outcomes",
    }


def fit_source_interactions(records):
    """Average actual source four-corner contrasts; exclude validation/application.

    Conditional keys are fixed before seeing results: frame, trigger actor,
    and report incidence. Family means, not duplicated rows, supply the unit.
    """
    source = [r for r in records if r["role"] == "source"]
    if not source:
        raise ValueError("joint source observations required")
    groups = {}
    for r in source:
        key = (r["frame"], r["actor"], r["incidence"])
        groups.setdefault(key, []).append(r)
    fitted = {}
    for key, rows in groups.items():
        if len({r["family"] for r in rows}) != len(rows):
            raise ValueError("source family duplicated within a conditional unit")
        y = np.array([r["interaction"] for r in rows], float)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite source interaction")
        fitted[key] = {
            "mean": float(y.mean()),
            "se": float(y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else None,
            "families": len(y),
        }
    return {
        "conditional": fitted,
        "global": float(np.mean([r["interaction"] for r in source])),
        "source_rows": len(source),
        "source_families": len({r["family"] for r in source}),
    }


def paired_family_interval(families, improvement, seed=1010, draws=2000):
    """Return a paired family bootstrap for squared-error improvement."""
    f, d = np.asarray(families), np.asarray(improvement, float)
    if f.shape != d.shape or not len(d) or not np.isfinite(d).all():
        raise ValueError("aligned finite family contrasts required")
    means = np.array([d[f == key].mean() for key in np.unique(f)])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), (draws, len(means)))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()

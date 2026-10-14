#!/usr/bin/env python3
"""Decompose frozen target functionals against observed control responses."""

from __future__ import annotations

import numpy as np


def finite_matrix(values):
    """Require a nonempty finite matrix, without inferring semantic meaning."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("A nonempty finite matrix is required")
    return x


def control_design(rows):
    """Match two declared control assignments to neutral by task and seed."""
    keys = [(r["situation"], r["seed"], r["condition"]) for r in rows]
    controls = {}
    for i, row in enumerate(rows):
        if row["axis"] == "none":
            key = keys[i]
            if key in controls:
                raise ValueError("Duplicate control observation")
            controls[key] = i
    settings = sorted({(task, seed) for task, seed, _ in controls})
    if not settings:
        raise ValueError("No observed control settings")
    weights, metadata = [], []
    for task, seed in settings:
        for condition in ("nonpersona_newsletter", "nonpersona_spelling"):
            a = np.zeros(len(rows))
            try:
                a[controls[task, seed, condition]] = 1
                a[controls[task, seed, "neutral"]] = -1
            except KeyError as error:
                raise ValueError(
                    "An observed control/neutral pair is missing"
                ) from error
            weights.append(a)
            metadata.append({"situation": task, "seed": seed, "condition": condition})
    return np.asarray(weights), metadata


def response_projection(contrasts, coefficients, relative_tolerance=1e-10):
    """Split covectors into a finite empirical response span and its complement.

    The split is bookkeeping, not denoising or a causal nuisance assumption.
    The tolerance is a numerical rank rule, never selected on cold outcomes.
    """
    b = finite_matrix(contrasts)
    w = np.asarray(coefficients, dtype=float)
    if w.ndim < 1 or w.shape[-1] != b.shape[1] or not np.isfinite(w).all():
        raise ValueError("Finite covectors in the same feature space are required")
    if not 0 < relative_tolerance < 1:
        raise ValueError("A fixed positive relative rank tolerance is required")
    u, singular, vt = np.linalg.svd(b, full_matrices=False)
    keep = singular > relative_tolerance * singular[0] if singular[0] > 0 else []
    basis = vt[keep] if singular[0] > 0 else np.empty((0, b.shape[1]))
    coordinates = w @ basis.T
    shared = coordinates @ basis
    distinct = w - shared
    dual = (
        (coordinates / singular[keep]) @ u[:, keep].T
        if len(basis)
        else np.zeros((*w.shape[:-1], len(b)))
    )
    return {
        "basis": basis,
        "singular_values": singular,
        "parts": np.stack((shared, distinct), axis=-2),
        "dual": dual,
    }


def score_features(features, parts):
    """Apply two retained covector parts to numeric observations only."""
    x = finite_matrix(features)
    w = np.asarray(parts, dtype=float)
    if w.ndim != 4 or w.shape[-2:] != (2, x.shape[1]) or not np.isfinite(w).all():
        raise ValueError("Arm/target/two-part covectors must match the observations")
    return np.einsum("nf,akpf->nakp", x, w)


def target_couplings(parts):
    """Return the exact target Gram decomposition; do not add normalized cosines."""
    w = np.asarray(parts, dtype=float)
    if w.ndim != 4 or w.shape[2] != 2 or not np.isfinite(w).all():
        raise ValueError("Finite arm/target/two-part covectors required")
    full = w.sum(axis=2)
    return {
        "full": np.einsum("akf,alf->akl", full, full),
        "overlap": np.einsum("akf,alf->akl", w[:, :, 0], w[:, :, 0]),
        "complement": np.einsum("akf,alf->akl", w[:, :, 1], w[:, :, 1]),
        "cross": np.einsum("akf,alf->akl", w[:, :, 0], w[:, :, 1]),
    }


def coverage(contrasts, basis):
    """Describe finite response-span coverage without calling it information."""
    x = finite_matrix(contrasts)
    q = np.asarray(basis, dtype=float)
    if q.ndim != 2 or q.shape[1] != x.shape[1] or not np.isfinite(q).all():
        raise ValueError("A finite aligned basis is required")
    total = np.sum(x * x, axis=1)
    projected = np.sum((x @ q.T) ** 2, axis=1)
    fraction = np.full(len(x), np.nan)
    np.divide(projected, total, out=fraction, where=total > 0)
    return fraction

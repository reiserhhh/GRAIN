#!/usr/bin/env python3
"""Cold source-qualified GRAIN meanings and event-bound model action accounts."""

from __future__ import annotations

import numpy as np


def event_mass(cells, event_ids):
    """Construct an empirical law from distinct, addressed eight-cell events."""
    a = np.asarray(cells)
    if (
        a.ndim != 1
        or not len(a)
        or not np.issubdtype(a.dtype, np.integer)
        or not np.isin(a, np.arange(8)).all()
        or len(event_ids) != len(a)
        or len(set(event_ids)) != len(a)
    ):
        raise ValueError(
            "Unique addressed events in the fixed eight-cell family required"
        )
    return np.bincount(a, minlength=8) / len(a)


def fields(reference, probabilities, prior):
    """Retain source target fields, conditional parts and source-only projections."""
    r, psi, pi = (np.asarray(x, float) for x in (reference, probabilities, prior))
    if (
        r.shape != (8,)
        or psi.shape != (8, 5)
        or pi.shape != (5,)
        or not all(np.isfinite(x).all() for x in (r, psi, pi))
        or np.any(r <= 0)
        or not np.isclose(r.sum(), 1)
        or np.any((psi < 0) | (psi > 1))
        or np.any((pi <= 0) | (pi >= 1))
    ):
        raise ValueError(
            "Complete positive source support and five target fields required"
        )
    w = psi - pi
    if np.max(abs(r @ w)) > 2e-12:
        raise ValueError("Source field is not anchored to its own reference")
    condition = np.arange(8) // 4
    mu = np.stack(
        [
            r[condition == c] @ w[condition == c] / r[condition == c].sum()
            for c in (0, 1)
        ]
    )
    choice = mu[condition]
    within = w - choice
    unique = np.empty_like(w)
    for q in range(5):
        others = np.delete(w, q, axis=1)
        design = np.sqrt(r)[:, None] * others
        coef = np.linalg.lstsq(design, np.sqrt(r) * w[:, q], rcond=None)[0]
        unique[:, q] = w[:, q] - others @ coef
    return {
        "full": w,
        "choice": choice,
        "within": within,
        "unique": unique,
        "shared": w - unique,
    }


def cold_scores(masses, reference, source_fields):
    """Score evidence laws at fixed SOURCE units, without labels or model answers."""
    p, r = np.asarray(masses, float), np.asarray(reference, float)
    if (
        p.ndim != 2
        or p.shape[1] != 8
        or r.shape != (8,)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=2e-12)
    ):
        raise ValueError(
            "Complete evidence laws required, not zero-filled missing observations"
        )
    return {k: (p - r) @ v for k, v in source_fields.items()}


def gram(values, weight=None):
    """Describe signed geometry; no spectrum is selected to improve an outcome."""
    x = np.asarray(values, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite row observations required")
    p = np.full(len(x), 1 / len(x)) if weight is None else np.asarray(weight, float)
    if p.shape != (len(x),) or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Declared positive observation weights required")
    g = x.T @ (p[:, None] * x)
    norms = np.sqrt(np.diag(g))
    denom = norms[:, None] * norms[None, :]
    cosine = np.divide(g, denom, out=np.zeros_like(g), where=denom > 0)
    eig = np.linalg.eigvalsh(g)
    tol = np.finfo(float).eps * max(x.shape) * max(float(eig[-1]), 0)
    trace = float(np.trace(g))
    effective = trace**2 / float(np.sum(g * g)) if trace > 0 else None
    return {
        "gram": g,
        "cosine": cosine,
        "eigenvalues": eig,
        "rank": int(np.sum(eig > tol)),
        "effective_rank": effective,
    }


def bridge(edge_scores, actions):
    """Contract all five meanings with reader/source actions in unchanged units."""
    x, t = np.asarray(edge_scores, float), np.asarray(actions, float)
    if (
        x.ndim != 2
        or x.shape[1] != 5
        or t.shape != (len(x), 5, 5)
        or not np.isfinite(x).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("Matched edge x meaning and edge x reader x source required")
    contact = np.einsum("et,eqr->tqr", x, t) / len(x)
    den = (
        np.sqrt(np.mean(x**2, axis=0))[:, None, None]
        * np.sqrt(np.mean(t**2, axis=0))[None]
    )
    normalized = np.divide(contact, den, out=np.zeros_like(contact), where=den > 0)
    return {"contact": contact, "cosine": normalized, "defined": den > 0}

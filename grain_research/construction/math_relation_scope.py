#!/usr/bin/env python3
"""Relation-indexed finite observations, not a semantic classifier or trait fit."""

from __future__ import annotations

import itertools

import numpy as np


def design_basis():
    """Return the fixed four-bit corners and their complete Walsh basis."""
    points = np.asarray(list(itertools.product((-1, 1), repeat=4)), float)
    masks = np.asarray(list(itertools.product((0, 1), repeat=4)), int)
    basis = np.prod(points[None] ** masks[:, None], axis=2)
    names = [
        "".join(c for c, b in zip("xygh", mask, strict=True) if b) or "1"
        for mask in masks
    ]
    return points, basis, names


def coefficients(fields):
    """Resolve all designed corner interactions without selecting coordinates."""
    f = np.asarray(fields, float)
    if f.shape[0] != 16 or not np.isfinite(f).all():
        raise ValueError("Complete finite16-corner observations required")
    return np.tensordot(design_basis()[1] / 16, f, axes=(1, 0))


def scalar_account(reference, response):
    """Describe scalar proportionality via projection and an exterior-product norm."""
    u, v = np.asarray(reference, float).ravel(), np.asarray(response, float).ravel()
    if u.shape != v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Aligned finite response coordinates required")
    uu, vv, uv = float(u @ u), float(v @ v), float(u @ v)
    if uu <= 0 or vv <= 0:
        return {"reference_energy": uu, "response_energy": vv, "observed": False}
    multiplier = uv / uu
    remainder = v - multiplier * u
    residual_energy = float(remainder @ remainder)
    return {
        "observed": True,
        "reference_energy": uu,
        "response_energy": vv,
        "multiplier": multiplier,
        "cosine": uv / np.sqrt(uu * vv),
        "non_scalar_energy_fraction": residual_energy / vv,
        "wedge_squared": max(0.0, uu * vv - uv * uv),
        "identity_error": abs(uu * residual_energy - (uu * vv - uv * uv)),
    }


def product_account(old, new):
    """Keep all seven changes in selection, subject gate and content projection."""
    p, h, c = (np.asarray(a, float) for a in old)
    p1, h1, c1 = (np.asarray(a, float) for a in new)
    if any(a.shape != p.shape for a in (h, c, p1, h1, c1)):
        raise ValueError("Matched token/channel coordinates required")
    if not all(np.isfinite(a).all() for a in (p, h, c, p1, h1, c1)):
        raise ValueError("Finite product factors required")
    dp, dh, dc = p1 - p, h1 - h, c1 - c
    terms = {
        "selection": float(np.sum(dp * h * c)),
        "subject_gate": float(np.sum(p * dh * c)),
        "value_content": float(np.sum(p * h * dc)),
        "selection_gate": float(np.sum(dp * dh * c)),
        "selection_value": float(np.sum(dp * h * dc)),
        "gate_value": float(np.sum(p * dh * dc)),
        "triple": float(np.sum(dp * dh * dc)),
    }
    delta = float(np.sum(p1 * h1 * c1) - np.sum(p * h * c))
    return {**terms, "delta": delta, "identity_error": abs(delta - sum(terms.values()))}


def relation_laws(weights, epsilon=0.5):
    """Create laws with equal relation AND complete content/subject marginals."""
    w = np.asarray(weights, float)
    if w.ndim != 1 or not len(w) or abs(w.mean()) > 1e-12:
        raise ValueError("Finite centered relation weights required")
    if not np.isfinite(w).all() or not 0 < epsilon <= 1 or np.max(np.abs(w)) > 1:
        raise ValueError("Bounded weights and positive perturbation required")
    points = design_basis()[0]
    z = (points[:, 0] * points[:, 2] + points[:, 1] * points[:, 3]) / 2
    tilt = epsilon * w[:, None] * z[None]
    return (1 + tilt) / tilt.size, (1 - tilt) / tilt.size

#!/usr/bin/env python3
"""Finite history/event interaction and same-marginal composition identities."""

from __future__ import annotations

import itertools

import numpy as np


def design_basis():
    """Return all six-bit corners, characters and their event-group supports."""
    points = np.asarray(list(itertools.product((-1, 1), repeat=6)), float)
    masks = np.asarray(list(itertools.product((0, 1), repeat=6)), int)
    basis = np.prod(points[None] ** masks[:, None], axis=2)
    names = [
        "".join(c for c, bit in zip("xyghab", mask, strict=True) if bit) or "1"
        for mask in masks
    ]
    mixed = masks[:, (0, 2, 4)].any(1) & masks[:, (1, 3, 5)].any(1)
    return points, basis, names, mixed


def coefficients(fields):
    """Expand a complete 64-corner observation without truncation or fitting."""
    f = np.asarray(fields, float)
    if f.shape[0] != 64 or not np.isfinite(f).all():
        raise ValueError("Complete finite64-corner observations required")
    return np.tensordot(design_basis()[1] / 64, f, axes=(1, 0))


def composition_account(fields):
    """Resolve the unique design-orthogonal additive and mixed-event parts."""
    f = np.asarray(fields, float)
    coef = coefficients(f)
    _, basis, _, mixed = design_basis()
    residual = np.tensordot(basis[mixed].T, coef[mixed], axes=(1, 0))
    total = float(np.sum(coef[1:] ** 2))
    cross = float(np.sum(coef[mixed] ** 2))
    return {
        "nonconstant_energy": total,
        "cross_event_energy": cross,
        "cross_event_fraction": cross / total if total > 0 else 0.0,
        "cross_event_rms": float(np.sqrt(np.mean(residual**2))),
        "parseval_error": abs(float(np.mean(residual**2)) - cross / f[0].size),
    }


def author_laws(epsilon=0.5):
    """Keep both complete event marginals while tilting a past/current pairing."""
    if not 0 < epsilon <= 1:
        raise ValueError("Positive bounded epsilon required")
    x = design_basis()[0]
    tilt = epsilon * x[:, 4] * x[:, 1] * x[:, 3]
    return (1 + tilt) / 64, (1 - tilt) / 64


def reader_account(matrix):
    """Separate the fixed reader-battery mean without choosing a low rank."""
    a = np.asarray(matrix, float)
    if a.shape != (6, 6) or not np.isfinite(a).all():
        raise ValueError("Finite six-family/six-reader matrix required")
    common = np.broadcast_to(a.mean(1)[:, None], a.shape)
    specific = a - common
    full_energy = float(np.sum(a**2))
    common_energy = float(np.sum(common**2))
    specific_energy = float(np.sum(specific**2))
    return {
        "full_energy": full_energy,
        "battery_common_energy": common_energy,
        "reader_specific_energy": specific_energy,
        "identity_error": abs(full_energy - common_energy - specific_energy),
    }

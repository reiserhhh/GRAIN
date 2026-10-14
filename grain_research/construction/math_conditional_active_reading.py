#!/usr/bin/env python3
"""HM37 source-only finite-character laws in a frozen target-active coframe."""

from __future__ import annotations

import itertools

import numpy as np

CELLS = tuple(itertools.product(range(2), repeat=3))
LAWS = ("global", "node", "fiber", "full", "mismatched")
NODES = (0.0, 0.5, 1.0)


def characters():
    """All eight real characters of the supplied finite design group Z2^3."""
    bits = np.array(CELLS, dtype=int)
    return (-1.0) ** (bits @ bits.T)


def encode_cells(cells):
    """Complete orthogonal character transform of the first, eight-cell axis."""
    a = np.asarray(cells, float)
    if a.shape[0] != 8 or not np.isfinite(a).all():
        raise ValueError("eight finite cells required")
    return np.tensordot(characters().T / 8, a, axes=(1, 0))


def decode_characters(coefficients):
    """Invert the complete character table without selecting effects."""
    a = np.asarray(coefficients, float)
    if a.shape[0] != 8:
        raise ValueError("all eight characters required")
    return np.tensordot(characters(), a, axes=(1, 0))


def source_laws(coefficients, roles):
    """Build five declared means from two source wordings, not application scores."""
    a = np.asarray(coefficients, float)
    if a.ndim != 5 or a.shape[:4] != (2, 8, 3, 2) or not np.isfinite(a).all():
        raise ValueError("two wordings/eight cells/three nodes/two arms required")
    if list(roles) != ["source", "source"]:
        raise ValueError("application coefficients cannot enter source laws")
    cells = a.mean(axis=0)
    fourier = encode_cells(cells)
    full = decode_characters(fourier)
    laws = np.stack(
        (
            np.broadcast_to(cells.mean(axis=(0, 1)), cells.shape),
            np.broadcast_to(cells.mean(axis=0, keepdims=True), cells.shape),
            np.broadcast_to(cells.mean(axis=1, keepdims=True), cells.shape),
            full,
            full[np.arange(8) ^ 3],
        )
    )
    return {
        "cells": cells,
        "characters": fourier,
        "laws": laws,
        "wording_residual": a - cells,
    }


def interpolate(coefficients, t):
    """Predeclared quadratic Lagrange interpolation on the three original nodes."""
    a = np.asarray(coefficients, float)
    if a.shape[0] != 3 or not 0 <= t <= 1:
        raise ValueError("three nodes and t in [0,1] required")
    weights = np.array((2 * (t - 0.5) * (t - 1), -4 * t * (t - 1), 2 * t * (t - 0.5)))
    return np.tensordot(weights, a, axes=(0, 0))


def predict(laws, cell, t, h_norms, active_displacement):
    """Predict with only source laws and supplied cold inputs, never cold covectors."""
    a = np.asarray(laws, float)
    norms = np.asarray(h_norms, float)
    z = np.asarray(active_displacement, float)
    if a.shape[:4] != (5, 8, 3, 2) or norms.shape != (2,) or z.shape != (a.shape[-1],):
        raise ValueError("five complete laws and two physical H norms required")
    if not (norms > 0).all() or cell not in range(8):
        raise ValueError("positive physical norms and addressed cell required")
    coefficients = (
        np.stack([interpolate(law[cell], t) for law in a]) * norms[None, :, None]
    )
    return coefficients, np.einsum("lar,r->la", coefficients, z)


def error_metrics(triples):
    """Separate carrier error, reading error and signed cancellation in predictions."""
    a = np.asarray(triples, float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("actual full, actual supported, predicted triples required")
    actual, supported, predicted = a.T
    carrier = actual - supported
    reading = supported - predicted
    error = actual - predicted
    actual_ms = float(np.mean(actual * actual))
    error_ms, carrier_ms, reading_ms = (
        float(np.mean(x * x)) for x in (error, carrier, reading)
    )
    cross = float(2 * np.mean(carrier * reading))
    return {
        "count": len(a),
        "actual_rms": float(np.sqrt(actual_ms)),
        "predicted_rms": float(np.sqrt(np.mean(predicted * predicted))),
        "rmse": float(np.sqrt(error_ms)),
        "relative_rmse": float(np.sqrt(error_ms / actual_ms)) if actual_ms else None,
        "carrier_error_rms": float(np.sqrt(carrier_ms)),
        "reading_error_rms": float(np.sqrt(reading_ms)),
        "error_cross_mean": cross,
        "error_identity": abs(error_ms - carrier_ms - reading_ms - cross),
        "sign_agreement": int(np.sum(actual * predicted > 0)),
        "actual_nonzero": int(np.sum(actual != 0)),
    }

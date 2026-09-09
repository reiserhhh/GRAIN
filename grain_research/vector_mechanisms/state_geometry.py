#!/usr/bin/env python3
"""Reduced-state geometry and exact commutation with a frozen linear reader."""

from __future__ import annotations

import numpy as np

from .state_transport import raw_kernel


def residual_basis(model):
    """Recover the construction residual span; no evaluation state enters its fit."""
    x = model["train_x"]
    k = raw_kernel(x, x, 1)
    c = model["col1"]
    kc = k - c[None, :] - c[:, None] + model["grand1"]
    residual = (kc + float(model["alpha"]) * np.eye(len(x))) @ model["coef1"]
    gram = residual @ residual.T
    eigen, vectors = np.linalg.eigh(gram)
    keep = eigen > max(float(eigen.max()) * 1e-10, 1e-16)
    basis = (vectors[:, keep].T @ residual) / np.sqrt(eigen[keep])[:, None]
    if not np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-6):
        raise ValueError("Residual basis orthogonality failed")
    return basis


def decompose_state_error(model, basis, early, actual, estimated):
    """Split error into unreachable residual and within-span coefficient error.

    The reachable oracle uses actual later-layer states and is diagnostic only.
    """
    base = early + model["mean_r"]
    target_residual = actual - base
    reachable = (target_residual @ basis.T) @ basis
    oracle = base + reachable
    outside = actual - oracle
    inside = oracle - estimated
    total = actual - estimated
    norm = lambda v: float(v @ v)
    energy_error = abs(norm(total) - norm(outside) - norm(inside))
    return (
        oracle,
        outside,
        inside,
        {
            "total_squared": norm(total),
            "outside_squared": norm(outside),
            "inside_squared": norm(inside),
            "orthogonality_accounting_error": energy_error,
        },
    )


def linear_reader(function):
    """Return one matrix for the exact mean/configuration/final score components."""
    rows, width = function.p_ref.shape
    matrix = np.zeros(((rows + 1) * width, 3))
    matrix[:-width, 0] = np.tile(function.mean_coef / rows, (rows, 1)).ravel()
    matrix[:-width, 1] = function.config_coef.ravel()
    matrix[-width:, 2] = function.g_final
    reference = np.r_[function.p_ref.ravel(), function.q_ref]
    return matrix, -reference @ matrix


def compile_projection(model, function):
    """Contract state-output matrices offline, retaining exactly the same function."""
    matrix, bias = linear_reader(function)
    out = {
        k: model[k].copy()
        for k in ("mean_x", "scale", "train_x", "col1", "col2", "grand1", "grand2")
    }
    out.update(
        {
            "reader": matrix,
            "offset": bias + model["mean_r"] @ matrix,
            "coef1": model["coef1"] @ matrix,
            "coef2": model["coef2"] @ matrix,
            "anchor": np.asarray(function.anchor),
        }
    )
    return out


def score_compiled_projection(program, early, degree=1):
    """Compute the three effective projections without constructing full P/Q."""
    x = np.asarray(early, np.float64)
    if x.shape != program["mean_x"].shape:
        raise ValueError("Compiled early-state shape mismatch")
    z = (x - program["mean_x"]) / float(program["scale"])
    k = raw_kernel(z[None, :], program["train_x"], degree)[0]
    kc = k - k.mean() - program[f"col{degree}"] + program[f"grand{degree}"]
    parts = x @ program["reader"] + program["offset"] + kc @ program[f"coef{degree}"]
    return {
        "mean": float(parts[0]),
        "configuration": float(parts[1]),
        "final": float(parts[2]),
        "score": float(program["anchor"] + parts.sum()),
    }

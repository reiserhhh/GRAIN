#!/usr/bin/env python3
"""Exact derivative-component accounting for the frozen NEXT70 scalar readers."""

from __future__ import annotations

import numpy as np

from . import normalized_evidence as reader

COMPONENTS = ("local", "content", "selection", "logmass")
CHANGES = ("local_refit", "content", "selection", "logmass")


def densities(body, last, model):
    """Return L,C,S,M as T*dF/dz; terms need not be integrable individually."""
    z, p, phi = reader.base.design(body, last, model)
    size, rank, n = int(model["local_width"]), z.shape[1], len(z)
    coef = np.asarray(model["coef"], float)
    if size != len(phi) or coef.shape != (size + 16 * (rank + 1),):
        raise ValueError("Frozen local plus16 content/logmass coefficients required")
    state = reader.kernel_state(z, p, model)
    v, b = coef[size : size + 16 * rank].reshape(16, rank), coef[-16:]
    local = np.concatenate(
        [d @ coef[:size] for _, d in reader.base.density_blocks(z, p, last, model)]
    )
    if str(model["arm"]) == reader.ARMS[0]:
        # Direct local weights avoid introducing roundoff-only cross-token dependence.
        delta = np.column_stack((z, p))[:, None] - model["landmarks"][::4][None]
        weight = np.exp(
            -np.sum(delta**2, axis=2) / (2 * float(model["bandwidth"]) ** 2)
        )
        contrast = z[:, None, :]
    elif str(model["arm"]) == reader.ARMS[1]:
        weight = n * state["a"]
        contrast = z[:, None, :] - state["r"][None]
    else:
        raise ValueError("Unknown evidence reader")
    content = weight @ v
    value = np.sum(contrast * v[None], axis=2)
    selection = np.einsum("tk,tkr,tk->tr", weight, state["s"], value)
    mass = np.einsum("tk,tkr,k->tr", n * state["a"], state["s"], b)
    parts = np.stack((local, content, selection, mass))
    if not np.isfinite(parts).all():
        raise FloatingPointError("Nonfinite evidence component")
    return parts


def actions(parts, directions, model):
    """Contract every component with full/mean-centered GRAIN directions."""
    parts, directions = np.asarray(parts, float), np.asarray(directions, float)
    if parts.ndim != 3 or directions.ndim != 3 or parts.shape[1] != directions.shape[1]:
        raise ValueError(
            "Aligned component/token/coordinate and factor/token/hidden arrays required"
        )
    if (
        parts.shape[2] != model["basis"].shape[1]
        or directions.shape[2] != model["basis"].shape[0]
    ):
        raise ValueError("Source basis shape differs")
    projected = (directions @ model["basis"]) / model["scale"]
    full = np.einsum("ctr,ftr->cf", parts, projected) / parts.shape[1]
    allocation = (
        np.einsum("ctr,ftr->cf", parts, projected - projected.mean(1, keepdims=True))
        / parts.shape[1]
    )
    return full, allocation


def moments(target, parts):
    """Equal-coordinate moments; caller averages authors for variable-length fields."""
    y, p = np.asarray(target, float), np.asarray(parts, float)
    if p.ndim != y.ndim + 1 or p.shape[1:] != y.shape or y.size == 0:
        raise ValueError("Aligned target and leading component dimension required")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite account inputs required")
    y, p = y.ravel(), p.reshape(len(p), -1)
    return float(y @ y / y.size), (p @ y) / y.size, (p @ p.T) / y.size


def projection_account(energy, projection, gram):
    """Signed projections, all cross terms and total SSE against a zero baseline."""
    energy, projection, gram = (
        float(energy),
        np.asarray(projection, float),
        np.asarray(gram, float),
    )
    if energy <= 0 or gram.shape != (len(projection), len(projection)):
        raise ValueError("Positive target energy and square component Gram required")
    return {
        "target_energy": energy,
        "projection": projection / energy,
        "gram": gram / energy,
        "sse_over_zero": float((energy - 2 * projection.sum() + gram.sum()) / energy),
    }


def improvement_account(target, baseline, changes):
    """Symmetric algebraic error credits, not interventions or causal shares."""
    y, base, d = (
        np.asarray(target, float),
        np.asarray(baseline, float),
        np.asarray(changes, float),
    )
    if base.shape != y.shape:
        raise ValueError("Matched baseline shape required")
    energy, projection, gram = moments(base - y, d)
    if energy <= 0:
        raise ValueError("Positive baseline error required")
    total = d.sum(0)
    new_error = float(np.mean((base + total - y) ** 2))
    credits = -(2 * projection + gram.sum(1)) / energy
    return {
        "baseline_error": energy,
        "new_error": new_error,
        "linear_terms": 2 * projection / energy,
        "gram": gram / energy,
        "credits": credits,
        "relative_improvement": 1 - new_error / energy,
        "closure_error": float(abs(credits.sum() - (1 - new_error / energy))),
    }

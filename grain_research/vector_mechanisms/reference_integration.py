#!/usr/bin/env python3
"""Reference-conditioned score profiles and anchor-preserving response kernels.

These are measurement interfaces, not a declaration that a model score is a
human trait. Reference moments use source authors only; kernel constraints
preserve specified source points rather than unseen authors.
"""

from __future__ import annotations

import numpy as np

from .conditional_models import fit_response_kernel, predict_response_kernel


def centered_cross_kernel(model, raw_cross):
    """Apply the source-fitted linear kernel transform without batch centering."""
    raw = np.asarray(raw_cross, float)[:, model["indices"]]
    if not np.isfinite(raw).all():
        raise ValueError("Finite cross-kernel required")
    k = (raw - raw.mean(1)[:, None] - model["raw_col"] + model["raw_grand"]) / model[
        "scale2"
    ]
    return k - k.mean(1)[:, None] - model["col"] + model["grand"]


def fit_anchor_response(gram, targets, indices, pairs, anchors):
    """Fit a value model plus a source-anchor-null response correction.

    Keep CON's linear-kernel trace weight and sum-loss alpha=1. The correction
    has no free intercept, so it vanishes at retained anchors. A held anchor
    or edge is excluded just like every other held source observation.
    """
    indices = np.asarray(indices, int)
    base = fit_response_kernel(gram, targets, indices, pairs)
    secant = fit_response_kernel(gram, targets, indices, pairs, secant=True)
    positions = {int(i): j for j, i in enumerate(indices)}
    if len(set(anchors)) != len(anchors):
        raise ValueError("Duplicate anchors")
    aa = np.array([positions[int(i)] for i in anchors if int(i) in positions], int)
    if not len(aa):
        raise ValueError("At least one training anchor required")
    k = centered_cross_kernel(base, np.asarray(gram)[indices])
    k = (k + k.T) / 2
    anchor_inverse = np.linalg.pinv(k[np.ix_(aa, aa)], rcond=1e-10, hermitian=True)
    anchor_rows = k[aa]
    kp = k - k[:, aa] @ anchor_inverse @ anchor_rows
    kp = (kp + kp.T) / 2
    edges = [
        (positions[a], positions[b])
        for a, b in pairs
        if a in positions and b in positions
    ]
    d = np.zeros((len(edges), len(indices)))
    for row, (a, b) in enumerate(edges):
        d[row, a], d[row, b] = -1, 1
    metric = np.eye(len(indices))
    if len(edges):
        metric += float(secant["lambda"]) * len(indices) / len(edges) * d.T @ d
    residual = np.asarray(targets)[indices] - predict_response_kernel(
        base, np.asarray(gram)[indices]
    )
    coef = np.linalg.solve(kp + np.linalg.solve(metric, np.eye(len(indices))), residual)
    return base | {
        "correction_coef": coef,
        "anchor_positions": aa,
        "anchor_inverse": anchor_inverse,
        "anchor_rows": anchor_rows,
        "response_lambda": secant["lambda"],
        "anchor_kernel_max": np.asarray(np.max(np.abs(kp[aa]))),
        "projected_trace_ratio": np.asarray(np.trace(kp) / np.trace(k)),
        "projected_min_eigenvalue": np.asarray(np.linalg.eigvalsh(kp).min()),
    }


def predict_anchor_response(model, cross_gram):
    """Score new inputs with frozen coefficients, no IDs or evaluation outcomes."""
    k = centered_cross_kernel(model, cross_gram)
    kp = (
        k
        - k[:, model["anchor_positions"]]
        @ model["anchor_inverse"]
        @ model["anchor_rows"]
    )
    return predict_response_kernel(model, cross_gram) + kp @ model["correction_coef"]


def _cube(values):
    """Validate a complete author-by-condition-by-channel score panel."""
    x = np.asarray(values, float)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Finite complete author/condition/channel panel required")
    return x


def _weights(weights, n):
    """Require an explicitly normalized common opportunity distribution."""
    w = np.asarray(weights, float)
    if (
        w.shape != (n,)
        or not np.isfinite(w).all()
        or np.any(w <= 0)
        or not np.isclose(w.sum(), 1)
    ):
        raise ValueError("Positive normalized condition weights required")
    return w


def fit_condition_reference(values, weights, *, reference_id):
    """Freeze condition means and author-level scales on an explicit reference."""
    x = _cube(values)
    w = _weights(weights, x.shape[1])
    if len(x) < 2 or not reference_id:
        raise ValueError("Two reference authors and reference identity required")
    b = x.mean(0)
    a = np.einsum("acq,c->aq", x - b, w)
    scale = a.std(0)
    # Constant channels stay reconstructible but have no standardized position.
    return {
        "condition_mean": b,
        "weights": w,
        "level_scale": scale,
        "reference_id": reference_id,
        "reference_authors": len(x),
    }


def condition_profile(values, reference):
    """Keep position and conditional shape separate; reconstruct every score."""
    x = _cube(values)
    b, w = reference["condition_mean"], reference["weights"]
    if x.shape[1:] != b.shape:
        raise ValueError("Condition/channel coordinate mismatch")
    offset = x - b
    level = np.einsum("acq,c->aq", offset, w)
    shape = offset - level[:, None, :]
    scale = reference["level_scale"]
    supported = scale > 1e-12
    standardized = np.full_like(level, np.nan)
    np.divide(level, scale, out=standardized, where=supported)
    return {
        "level": level,
        "response_shape": shape,
        "standardized_level": standardized,
        "standardization_supported": supported,
        "reconstruction_max_error": float(
            np.max(np.abs(b + level[:, None] + shape - x))
        ),
        "shape_weighted_mean_max": float(
            np.max(np.abs(np.einsum("acq,c->aq", shape, w)))
        ),
    }


def selection_response_decomposition(
    person, reference, observed_weights, reference_weights
):
    """Exactly separate response, exposure/choice, and their interaction.

    Inputs are condition-by-channel means and normalized condition frequencies.
    Without opportunity data the second component is exposure, not free choice.
    """
    m, b = np.asarray(person, float), np.asarray(reference, float)
    if (
        m.ndim != 2
        or m.shape != b.shape
        or not np.isfinite(m).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Aligned finite conditional means required")
    p, q = (np.asarray(v, float) for v in (observed_weights, reference_weights))
    for w in (p, q):
        if (
            w.shape != (len(m),)
            or not np.isfinite(w).all()
            or np.any(w < 0)
            or not np.isclose(w.sum(), 1)
        ):
            raise ValueError("Normalized nonnegative condition distributions required")
    response, choice, interaction = q @ (m - b), (p - q) @ b, (p - q) @ (m - b)
    return {
        "response": response,
        "choice_or_exposure": choice,
        "interaction": interaction,
        "total": p @ m - q @ b,
        "reconstruction_error": float(
            np.max(np.abs(p @ m - q @ b - response - choice - interaction))
        ),
    }

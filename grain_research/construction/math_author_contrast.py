#!/usr/bin/env python3
"""Reference-relative author contrasts and fixed observation-function accounts."""

from __future__ import annotations

import numpy as np


def probability(values):
    """Validate positive full-support reference probabilities without renormalizing."""
    p = np.asarray(values, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("strictly positive finite reference required")
    if not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("reference must sum to one")
    return p


def norm2(f, reference):
    """Compute the squared L2 norm in a declared reference population."""
    p = probability(reference)
    f = np.asarray(f, float)
    if f.shape != p.shape or not np.isfinite(f).all():
        raise ValueError("one finite contrast per reference event required")
    return float(p @ (f * f))


def condition_account(f, reference, conditions):
    """Keep condition-choice and within-condition evidence fields orthogonally."""
    p = probability(reference)
    f, c = np.asarray(f, float), np.asarray(conditions)
    norm2(f, p)
    if c.shape != p.shape:
        raise ValueError("condition/event alignment required")
    choice = np.zeros_like(f)
    for value in np.unique(c):
        mask = c == value
        choice[mask] = p[mask] @ f[mask] / p[mask].sum()
    within = f - choice
    return {
        "choice": choice,
        "within": within,
        "total_norm2": norm2(f, p),
        "choice_norm2": norm2(choice, p),
        "within_norm2": norm2(within, p),
        "cross": float(p @ (choice * within)),
    }


def channel_account(f, reference, channel):
    """Disintegrate a contrast through a known shared finite Markov channel."""
    p = probability(reference)
    f, k = np.asarray(f, float), np.asarray(channel, float)
    norm2(f, p)
    if (
        k.ndim != 2
        or k.shape[0] != len(p)
        or not np.isfinite(k).all()
        or np.any(k < 0)
        or not np.allclose(k.sum(1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("registered row-stochastic channel required")
    output_reference = p @ k
    positive = output_reference > 0
    out = np.zeros(k.shape[1])
    out[positive] = (p * f) @ k[:, positive] / output_reference[positive]
    total = norm2(f, p)
    visible = float(output_reference @ (out * out))
    lost = float(np.sum(p[:, None] * k * (f[:, None] - out) ** 2))
    return {
        "output_field": out,
        "output_reference": output_reference,
        "total_norm2": total,
        "visible_norm2": visible,
        "lost_norm2": lost,
        "closure_error": abs(total - visible - lost),
    }


def source_projection(features, reference):
    """Freeze the weighted source span using only a fixed machine-rank rule."""
    p = probability(reference)
    b = np.asarray(features, float)
    if b.ndim != 2 or len(b) != len(p) or not np.isfinite(b).all():
        raise ValueError("finite registered source feature matrix required")
    centered = b - p @ b
    scale = np.sqrt(p @ (centered * centered))
    scale = np.where(scale > 1e-12, scale, 1)
    normalized = centered / scale
    a = np.sqrt(p[:, None]) * normalized
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    tolerance = (
        100 * max(a.shape) * np.finfo(float).eps * singular[0] if len(singular) else 0
    )
    keep = singular > tolerance
    return {
        "basis": u[:, keep],
        "rank": int(keep.sum()),
        "singular": singular,
        "tolerance": float(tolerance),
        "weighted_source": a,
        "normalized_source": normalized,
        "source_center": p @ b,
        "source_scale": scale,
        "right": vt[keep],
    }


def project(values, reference, projection):
    """Project source functions, not application author labels or reader outputs."""
    p = probability(reference)
    f = np.asarray(values, float)
    if f.ndim == 1:
        norm2(f, p)
        weighted = np.sqrt(p) * f
        return projection["basis"] @ (projection["basis"].T @ weighted) / np.sqrt(p)
    if f.ndim != 2 or len(f) != len(p) or not np.isfinite(f).all():
        raise ValueError("aligned finite function columns required")
    weighted = np.sqrt(p[:, None]) * f
    return (projection["basis"] @ (projection["basis"].T @ weighted)) / np.sqrt(
        p[:, None]
    )


def target_account(f, targets, reference, projection):
    """Return exact visible/lost target contributions for a declared author contrast."""
    p = probability(reference)
    visible = project(f, p, projection)
    tv = project(targets, p, projection)
    lost, tl = np.asarray(f) - visible, np.asarray(targets) - tv
    truth = (p * f) @ targets
    retained, remainder = (p * visible) @ tv, (p * lost) @ tl
    return {
        "total_norm2": norm2(f, p),
        "visible_norm2": norm2(visible, p),
        "lost_norm2": norm2(lost, p),
        "norm_closure_error": abs(norm2(f, p) - norm2(visible, p) - norm2(lost, p)),
        "truth": truth,
        "visible_target": retained,
        "lost_target": remainder,
        "target_closure_error": float(np.max(abs(truth - retained - remainder))),
    }


def cold_score(compiled, event_ids):
    """Score only legitimate sampled events using the frozen source function table."""
    values = np.asarray(compiled, float)
    ids = np.asarray(event_ids)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("finite compiled source table required")
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("integer event addresses required")
    if np.any((ids < 0) | (ids >= len(values))):
        raise ValueError("event outside registered source vocabulary")
    if not len(ids):
        return {"status": "UNOBSERVED", "scores": None, "n": 0}
    observed = values[ids]
    se = observed.std(0, ddof=1) / np.sqrt(len(ids)) if len(ids) > 1 else None
    return {
        "status": "OBSERVED_FINITE_VOCABULARY",
        "scores": observed.mean(0).tolist(),
        "sampling_se_iid_simulation": se.tolist() if se is not None else None,
        "n": len(ids),
    }

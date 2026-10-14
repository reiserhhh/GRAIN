#!/usr/bin/env python3
"""SOURCE target-evidence association states and finite sufficiency witnesses."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def validate(p, y, w=None):
    """Validate a finite source law without smoothing or reading author names."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if (
        p.ndim != 2
        or y.ndim != 2
        or len(p) != len(y)
        or len(p) < 2
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=2e-12)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError(
            "Finite normalized source laws and aligned binary targets required"
        )
    w = np.full(len(p), 1 / len(p)) if w is None else np.asarray(w, float)
    if (
        w.shape != (len(p),)
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or not np.isclose(w.sum(), 1, rtol=0, atol=2e-12)
    ):
        raise ValueError("A normalized nonnegative source weight is required")
    return p, y, w


def state(p, y, w=None):
    """Compile target means, event law and their conditional association table."""
    p, y, w = validate(p, y, w)
    mu, pi = w @ y, w @ p
    joint = (y * w[:, None]).T @ p
    gamma = joint - mu[:, None] * pi
    supported = pi > 0
    after = np.full_like(joint, np.nan)
    after[:, supported] = mu[:, None] + gamma[:, supported] / pi[supported]
    return {
        "mu": mu,
        "pi": pi,
        "joint": joint,
        "gamma": gamma,
        "after": after,
        "supported": supported,
    }


def update(p, w, cell):
    """Condition on one new, qualified event; impossible evidence is an error."""
    p, w = np.asarray(p, float), np.asarray(w, float)
    if not isinstance(cell, (int, np.integer)) or not 0 <= cell < p.shape[1]:
        raise ValueError("In-range integer event cell required")
    mass = w * p[:, cell]
    probability = mass.sum()
    if probability <= 0 or not np.isfinite(probability):
        raise ValueError("New event has zero source support; no smoothing rescue")
    return mass / probability


def replay(p, y, events):
    """Replay distinct event IDs; exact aliases contribute no extra likelihood."""
    p, y, w = validate(p, y)
    seen, trajectory = {}, []
    for event in events:
        key, cell = event["event_id"], event["cell"]
        if not isinstance(key, str) or not key:
            raise ValueError("Explicit event identity is required")
        if key in seen:
            if seen[key] != cell:
                raise ValueError(
                    "Conflicting duplicate event; explicit revision required"
                )
            continue
        before = state(p, y, w)
        w = update(p, w, cell)
        after = state(p, y, w)
        trajectory.append((key, cell, before, after))
        seen[key] = cell
    return w, trajectory


def constraints(p, y, order):
    """Preserve full target profiles, event marginals and stated update order."""
    p, y, _ = validate(p, y)
    profiles, ids = np.unique(y, axis=0, return_inverse=True)
    indicators = np.eye(len(profiles))[ids]
    columns = [indicators, p]
    if order == 1:
        columns.extend(
            [
                (indicators[:, :, None] * p[:, None, :]).reshape(len(p), -1),
                np.column_stack(
                    [
                        p[:, a] * p[:, b]
                        for a in range(p.shape[1])
                        for b in range(a, p.shape[1])
                    ]
                ),
            ]
        )
    elif order != 0:
        raise ValueError("Only registered constraint orders0 and1 are supported")
    return np.column_stack(columns), profiles


def column_basis(matrix):
    """Get the declared numerical column span once, without tuning its rank."""
    matrix = np.asarray(matrix, float)
    q, r, _ = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    tolerance = max(matrix.shape) * np.finfo(float).eps * diagonal[0]
    rank = int(np.count_nonzero(diagonal > tolerance))
    return q[:, :rank], rank, float(tolerance)


def witness(p, y, basis, target, cells, fraction=0.5, tolerance=1e-10):
    """Construct positive state-matched laws, retaining absent witnesses too."""
    p, y, base = validate(p, y)
    likelihood = np.prod(p[:, cells], axis=1)
    functional = y[:, target] * likelihood
    residual = functional - basis @ (basis.T @ functional)
    relative = np.linalg.norm(residual) / max(np.linalg.norm(functional), 1e-300)
    if relative <= tolerance:
        return {
            "present": False,
            "relative_residual": float(relative),
            "plus": base.copy(),
            "minus": base.copy(),
            "gap": 0.0,
        }
    if not 0 < fraction < 1:
        raise ValueError("A fixed fraction strictly between0 and1 is required")
    eta = fraction / (len(p) * np.max(np.abs(residual)))
    plus, minus = base + eta * residual, base - eta * residual
    denominator_plus, denominator_minus = plus @ likelihood, minus @ likelihood
    if min(denominator_plus, denominator_minus) <= 0:
        raise ValueError("Unsupported witness endpoint")
    gap = plus @ functional / denominator_plus - minus @ functional / denominator_minus
    return {
        "present": True,
        "relative_residual": float(relative),
        "plus": plus,
        "minus": minus,
        "gap": float(gap),
    }


def law_quotient(p, event_count):
    """Identify identical finite laws exactly through their integer count vectors."""
    p = np.asarray(p, float)
    integer = np.rint(p * event_count).astype(np.int64)
    if not np.array_equal(integer / event_count, p):
        raise ValueError("Source probabilities do not equal declared integer counts")
    laws, ids, sizes = np.unique(
        integer, axis=0, return_inverse=True, return_counts=True
    )
    return {
        "count_vectors": laws,
        "class_ids": ids,
        "sizes": sizes,
        "closed_observation_algebra_dimension": len(laws),
    }

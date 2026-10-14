#!/usr/bin/env python3
"""Finite response geometry and author-law ambiguity, not psychological truth."""

from __future__ import annotations

import itertools

import numpy as np


def sectors(values):
    """Keep all signed actor/focus sectors and own-person observations."""
    x = np.asarray(values, float)
    if x.ndim != 6 or x.shape[2:5] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError(
            "Finite family/condition/polarity/owner/focus/feature field required"
        )
    gain = (x[:, :, 1] - x[:, :, 0]) / 2
    own, other = gain[:, :, 1], gain[:, :, 0]
    even, odd = (own + other) / 2, (own - other) / 2
    return {
        "own_A": own[:, :, 0],
        "own_B": own[:, :, 1],
        "own_focus_even": own.mean(2),
        "own_focus_odd": (own[:, :, 0] - own[:, :, 1]) / 2,
        "actor_even_focus_even": even.mean(2),
        "actor_even_focus_odd": (even[:, :, 0] - even[:, :, 1]) / 2,
        "actor_odd_focus_even": odd.mean(2),
        "actor_odd_focus_odd": (odd[:, :, 0] - odd[:, :, 1]) / 2,
    }


def matrix(field, indexed=False):
    """Return a column observation map, optionally retaining condition blocks."""
    f = np.asarray(field, float)
    if f.ndim != 3 or not np.isfinite(f).all():
        raise ValueError("Finite family/condition/feature field required")
    families, conditions, features = f.shape
    if not indexed:
        return f.reshape(families * conditions, features).T
    r = np.zeros((conditions * features, families * conditions))
    for family, condition in itertools.product(range(families), range(conditions)):
        r[
            condition * features : (condition + 1) * features,
            family * conditions + condition,
        ] = f[family, condition]
    return r


def spectrum(columns):
    """Report complete singular values with machine rank, without rank tuning."""
    a = np.asarray(columns, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Finite observation matrix required")
    s = np.linalg.svd(a, compute_uv=False)
    tol = np.finfo(float).eps * max(a.shape) * (s[0] if len(s) else 0)
    energy = s * s
    return {
        "rank": int(np.sum(s > tol)),
        "singular_values": s.tolist(),
        "machine_rank_tolerance": float(tol),
        "participation_rank": float(energy.sum() ** 2 / (energy @ energy))
        if np.any(energy)
        else 0.0,
        "trace": float(energy.sum()),
    }


def correspondence(source, target):
    """Compare all family permutations and response shape, preserving zero cases."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 3 or x.shape[0] != 6:
        raise ValueError("Aligned six-family fields required")
    a, b = matrix(x), matrix(y)
    g, h = a.T @ a, b.T @ b
    norm = np.linalg.norm(g) * np.linalg.norm(h)
    result = {
        "source_spectrum": spectrum(a),
        "target_spectrum": spectrum(b),
        "gram_alignment": float(np.sum(g * h) / norm) if norm else None,
        "relative_action_difference": float(np.linalg.norm(b - a) / np.linalg.norm(a))
        if np.linalg.norm(a)
        else None,
    }
    nx, ny = np.linalg.norm(x, axis=2), np.linalg.norm(y, axis=2)
    if np.any(nx == 0) or np.any(ny == 0):
        return result | {
            "direction_status": "UNAVAILABLE_ZERO_ACTION",
            "matching_rank": None,
        }
    cross = np.einsum("fcq,gcq->cfg", x / nx[..., None], y / ny[..., None])
    perms = list(itertools.permutations(range(6)))
    values = np.array(
        [
            np.mean([cross[c, f, p[f]] for c in range(x.shape[1]) for f in range(6)])
            for p in perms
        ]
    )
    diagonal = np.stack([cross[c].diagonal() for c in range(x.shape[1])], 1)
    actual = float(diagonal.mean())
    return result | {
        "direction_status": "OBSERVED",
        "matched_mean_cosine": actual,
        "per_family_condition_cosine": diagonal.tolist(),
        "matching_rank": int(np.sum(values >= actual - 1e-12)),
        "permutations": len(values),
        "permutation_mean": float(values.mean()),
        "matched_minus_permutation_mean": actual - float(values.mean()),
    }


def ambiguous_law(response, amplitude=0.4):
    """Construct one source-only null direction with equal global polarity margin."""
    r = np.asarray(response, float)
    if r.ndim != 2 or not np.isfinite(r).all() or not 0 < amplitude < 1:
        raise ValueError("Finite response and valid amplitude required")
    constrained = np.vstack((r, np.ones(r.shape[1])))
    _, s, vt = np.linalg.svd(constrained, full_matrices=True)
    tol = np.finfo(float).eps * max(constrained.shape) * s[0]
    rank = int(np.sum(s > tol))
    basis = vt[rank:].T
    if not basis.shape[1]:
        return {"status": "NO_SOURCE_AMBIGUITY", "constraint_rank": rank}, np.zeros(
            r.shape[1]
        )
    projector = basis @ basis.T
    index = int(np.argmax(np.diag(projector)))
    direction = projector[:, index]
    theta = amplitude * direction / np.max(abs(direction))
    return {
        "status": "SOURCE_ONLY_AMBIGUOUS_LAW",
        "constraint_rank": rank,
        "null_dimension": basis.shape[1],
        "basis_index": index,
        "constraint_error": float(np.max(abs(constrained @ theta))),
    }, theta


def law_probabilities(theta):
    """Keep family/condition masses and global polarity equal for paired laws."""
    t = np.asarray(theta, float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.max(abs(t)) >= 1:
        raise ValueError("Interior finite author coefficients required")
    signs = np.array([-1, 1])
    return (1 + t[:, None] * signs[None]) / (2 * len(t))


def observation_bundle(own_uniform, own_weighted, own_values):
    """Keep aggregate, joint and condition-indexed observations as distinct maps."""
    u, r = matrix(own_uniform), matrix(own_weighted)
    return {
        "weighted": r,
        "uniform": u,
        "joint_uniform_weighted": np.vstack((u, r)),
        "condition_indexed_weighted": matrix(own_weighted, indexed=True),
        "value_vectors": matrix(own_values),
    }


def contrast_report(maps, theta):
    """Evaluate the same fixed law on every retained observation channel."""
    t = np.asarray(theta, float)
    return {
        name: spectrum(r)
        | {
            "contrast_norm": float(np.linalg.norm(2 * r @ t / len(t))),
            "contrast": (2 * r @ t / len(t)).tolist(),
        }
        for name, r in maps.items()
    }

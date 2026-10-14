#!/usr/bin/env python3
"""Higher-order evidence diagnostics with exactly uninformative lower marginals."""

from __future__ import annotations

import itertools

import numpy as np

POINTS = np.asarray(list(itertools.product((-1, 1), repeat=3)), dtype=int)
FIELD_POINTS = np.asarray(list(itertools.product((-1, 1), repeat=4)), dtype=int)


def population_counts(reference):
    """Return positive H/L counts for eight three-occasion configurations."""
    if reference not in (-1, 1):
        raise ValueError("A signed reference law is required")
    parity = np.prod(POINTS, axis=1)
    high = np.where(reference * parity == 1, 20, 5)
    return np.stack((high, 25 - high))


def posterior(choices, reference, observed=(0, 1, 2)):
    """Marginalize unobserved occasions, without falsely assuming independence."""
    b = np.asarray(choices)
    indices = tuple(observed)
    if (
        b.shape != (3,)
        or not np.isin(b, (-1, 1)).all()
        or len(set(indices)) != len(indices)
        or not all(i in range(3) for i in indices)
    ):
        raise ValueError("Three binary choices and distinct valid indices required")
    compatible = np.ones(8, dtype=bool)
    for i in indices:
        compatible &= POINTS[:, i] == b[i]
    counts = population_counts(reference)[:, compatible].sum(axis=1)
    return float(counts[0] / counts.sum())


def lower_marginals(reference):
    """Enumerate all conditional one/two-occasion tables, including total mass."""
    counts = population_counts(reference).astype(float) / 100
    result = {}
    for degree in range(3):
        for indices in itertools.combinations(range(3), degree):
            rows = []
            for values in itertools.product((-1, 1), repeat=degree):
                mask = np.ones(8, dtype=bool)
                for i, value in zip(indices, values, strict=True):
                    mask &= POINTS[:, i] == value
                rows.append(counts[:, mask].sum(axis=1))
            result[indices] = np.asarray(rows).T
    return result


def character_matrix():
    """Return the full classical character basis of the four-bit group."""
    return np.stack(
        [
            np.prod(FIELD_POINTS[:, [i for i in range(4) if mask & (1 << i)]], axis=1)
            for mask in range(16)
        ],
        axis=1,
    ).astype(float)


def coefficients(field):
    """Keep every effect, not only the anticipated b1*b2*b3*reference term."""
    x = np.asarray(field, dtype=float)
    if x.shape[:4] != (2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("A complete finite four-factor field is required")
    flat = x.reshape((16, *x.shape[4:]))
    return np.tensordot(character_matrix().T / 16, flat, axes=(1, 0))


def reconstruct(beta):
    """Rebuild all cells from their complete character decomposition."""
    b = np.asarray(beta, dtype=float)
    if b.shape[0] != 16 or not np.isfinite(b).all():
        raise ValueError("All sixteen finite coefficients required")
    flat = np.tensordot(character_matrix(), b, axes=(1, 0))
    return flat.reshape((2, 2, 2, 2, *b.shape[1:]))


def parity_contrasts(field):
    """Compare positive/negative complete configurations within each reference."""
    x = np.asarray(field, dtype=float)
    if x.shape[:4] != (2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete behavior/reference field required")
    weights = np.prod(POINTS, axis=1)
    return np.tensordot(weights / 4, x.reshape((8, 2, *x.shape[4:])), axes=(0, 0))


def effect_comparison(actual, reference):
    """Report signed effect alignment and size; zero targets remain explicit."""
    a, b = np.asarray(actual, float).ravel(), np.asarray(reference, float).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Compatible finite effects required")
    aa, bb, ab = a @ a, b @ b, a @ b
    return {
        "actual_rms": float(np.sqrt(aa / len(a))),
        "reference_rms": float(np.sqrt(bb / len(b))),
        "cosine": float(ab / np.sqrt(aa * bb)) if aa > 0 and bb > 0 else None,
        "signed_projection": float(ab / bb) if bb > 0 else None,
        "relative_error": float(np.linalg.norm(a - b) / np.sqrt(bb))
        if bb > 0
        else None,
    }

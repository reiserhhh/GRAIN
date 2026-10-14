#!/usr/bin/env python3
"""Lossless c/b/o/person/focus accounts, not a trained observer correction."""

from __future__ import annotations

import itertools

import numpy as np

SIGNS = np.array(list(itertools.product((-1, 1), repeat=5)))
CHARACTERS = np.array(
    [
        [np.prod([row[j] if mask & (1 << j) else 1 for j in range(5)]) for row in SIGNS]
        for mask in range(32)
    ],
    dtype=float,
)


def decompose(cells):
    """Keep all32 characters of complete c/b/o/p/r support and trailing units."""
    x = np.asarray(cells, float)
    if x.shape[:5] != (2,) * 5 or not np.isfinite(x).all():
        raise ValueError("Complete finite c/b/o/p/r support required")
    return (CHARACTERS @ x.reshape(32, -1) / 32).reshape(32, *x.shape[5:])


def reconstruct(beta):
    """Reconstruct every cell; no component is declared noise or discarded."""
    z = np.asarray(beta, float)
    if z.shape[0] != 32 or not np.isfinite(z).all():
        raise ValueError("All32 finite characters required")
    return (CHARACTERS.T @ z.reshape(32, -1)).reshape(*(2,) * 5, *z.shape[1:])


def qualified_effect(beta, condition, focus):
    """Resolve bo effects into query-person distinction and common focus response."""
    if condition not in (-1, 1) or focus not in (-1, 1):
        raise ValueError("Declared condition/focus signs required")
    z = np.asarray(beta, float)
    if z.shape[0] != 32 or not np.isfinite(z).all():
        raise ValueError("Complete finite character law required")
    return {
        "actor_difference": z[14]
        + condition * z[15]
        + focus * z[30]
        + condition * focus * z[31],
        "actor_common": z[6]
        + condition * z[7]
        + focus * z[22]
        + condition * focus * z[23],
    }


def positive_author_integral(cells, amplitude=0.4):
    """Integrate designed positive author contrasts separately for person/focus."""
    x = np.asarray(cells, float)
    if x.shape[:5] != (2,) * 5 or not np.isfinite(x).all() or not 0 < amplitude <= 1:
        raise ValueError("Complete support and legal positive-law amplitude required")
    out = np.zeros(x.shape[3:])
    for c, b, o in itertools.product(range(2), repeat=3):
        out += amplitude * (2 * b - 1) * (2 * o - 1) * x[c, b, o] / 24
    return out

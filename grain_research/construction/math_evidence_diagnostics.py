#!/usr/bin/env python3
"""Exact finite-coordinate attribution of a frozen multilinear score function."""

from itertools import permutations

import numpy as np


def coordinate_attribution(start, end, score):
    """Average all coordinate-replacement orders; preserve exact score change.

    This is a mathematical decomposition of the compiled program, not a causal
    intervention on an encoder, an attention head or a human psychological state.
    Signed contributions can cancel, so fractions need not lie in [0, 1].
    """
    a, b = np.asarray(start, float), np.asarray(end, float)
    if (
        a.ndim != 2
        or a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Aligned finite coordinate matrices required")
    if not 1 <= a.shape[1] <= 5:
        raise ValueError("Finite exact attribution limited to five coordinates")
    phi = np.zeros_like(a)
    count = 0
    for order in permutations(range(a.shape[1])):
        z = a.copy()
        previous = np.asarray(score(z), float)
        if previous.shape != (len(a),):
            raise ValueError("Score callback must return one scalar per row")
        for j in order:
            z[:, j] = b[:, j]
            current = np.asarray(score(z), float)
            phi[:, j] += current - previous
            previous = current
        count += 1
    return phi / count

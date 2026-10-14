#!/usr/bin/env python3
"""Finite author-distribution accounts, distinct from reader or trait accuracy."""

from __future__ import annotations

import itertools

import numpy as np

from .math_joint_law import mobius

FACTORS = ("choice", "expression", "law")
CONDITIONS = np.asarray(list(itertools.product((-1, 1), repeat=2)), float)
CASES = (
    "choice_f0",
    "choice_f1",
    "expression_f0",
    "expression_f1",
    "law_only",
    "mixed",
    "same_scores_f0",
    "same_scores_f1",
)


def profile(pi, mu):
    """Return positive joint x-by-condition probabilities in HM68 cell order."""
    pi, mu = np.asarray(pi, float), np.asarray(mu, float)
    if (
        pi.shape != (4,)
        or mu.shape != (4,)
        or not np.isfinite(pi).all()
        or not np.isfinite(mu).all()
        or np.any(pi <= 0)
        or np.any(abs(mu) >= 1)
        or abs(pi.sum() - 1) > 1e-12
    ):
        raise ValueError(
            "Four positive condition probabilities and interior means required"
        )
    return np.concatenate((pi * (1 - mu) / 2, pi * (1 + mu) / 2))


def disintegrate(p):
    """Keep composition and conditional expression; never fill absent conditions."""
    p = np.asarray(p, float)
    if (
        p.shape != (8,)
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or abs(p.sum() - 1) > 1e-12
    ):
        raise ValueError("Complete positive profile required for this finite protocol")
    table = p.reshape(2, 4)
    pi = table.sum(0)
    return pi, (table[1] - table[0]) / pi


def observe(p, law):
    """Integrate a profile using a declared observation law and its own units."""
    law = np.asarray(law, float)
    if law.ndim < 2 or law.shape[0] != 8 or not np.isfinite(law).all():
        raise ValueError("Complete eight-corner finite observation required")
    disintegrate(p)
    return np.tensordot(p, law, axes=(0, 0))


def equal_score_profiles(scores):
    """Construct an exact six-score fiber, not a training or precision search."""
    f = np.asarray(scores, float)
    if f.shape != (8, 6) or not np.isfinite(f).all():
        raise ValueError("HM68 eight-by-six scores required")
    matrix = np.column_stack((np.ones(8), f))
    u, s, _ = np.linalg.svd(matrix, full_matrices=True)
    tolerance = np.finfo(float).eps * max(matrix.shape) * s[0]
    rank = int(np.sum(s > tolerance))
    vector = u[:, rank]
    if vector[np.argmax(abs(vector))] < 0:
        vector = -vector
    vector = vector / np.max(abs(vector))
    a, b = np.full(8, 1 / 8) - 0.05 * vector, np.full(8, 1 / 8) + 0.05 * vector
    disintegrate(a)
    disintegrate(b)
    return a, b, rank


def case_profiles(case, score0, score1):
    """Build all fixed mechanistic author profiles without sampling a population."""
    if case not in CASES:
        raise ValueError("Declared case required")
    y, g = CONDITIONS.T
    pi0 = np.full(4, 0.25)
    pi1 = np.exp(0.6 * y - 0.2 * g + 0.15 * y * g)
    pi1 /= pi1.sum()
    mu0, mu1 = 0.2 * y - 0.1 * g + 0.15 * y * g, -0.1 * y + 0.2 * g - 0.35 * y * g
    if case.startswith("choice"):
        return profile(pi0, mu0), profile(pi1, mu0), int(case[-1]), int(case[-1]), None
    if case.startswith("expression"):
        return (
            profile(pi0, -0.5 * y * g),
            profile(pi0, 0.5 * y * g),
            int(case[-1]),
            int(case[-1]),
            None,
        )
    if case.startswith("same_scores"):
        family = int(case[-1])
        a, b, rank = equal_score_profiles((score0, score1)[family])
        return a, b, family, family, rank
    if case == "law_only":
        p = profile(pi0, mu0)
        return p, p.copy(), 0, 1, None
    return profile(pi0, mu0), profile(pi1, mu1), 0, 1, None


def accounts(p0, p1, law0, law1):
    """Save every hybrid, replacement order, interaction and symmetric account."""
    pi0, mu0 = disintegrate(p0)
    pi1, mu1 = disintegrate(p1)
    laws = (np.asarray(law0, float), np.asarray(law1, float))
    if laws[0].shape != laws[1].shape:
        raise ValueError("Common coded support/output units required")
    game = np.asarray(
        [
            observe(
                profile((pi0, pi1)[bool(m & 1)], (mu0, mu1)[bool(m & 2)]),
                laws[bool(m & 4)],
            )
            for m in range(8)
        ]
    )
    differences = mobius(game)
    shapley = np.zeros((3, *game.shape[1:]))
    for m in range(1, 8):
        for j in range(3):
            if m & (1 << j):
                shapley[j] += differences[m] / m.bit_count()
    sequences = []
    for order in itertools.permutations(range(3)):
        values, mask = np.zeros_like(shapley), 0
        for j in order:
            after = mask | (1 << j)
            values[j] = game[after] - game[mask]
            mask = after
        sequences.append(values)
    return {
        "game": game,
        "interactions": differences,
        "shapley": shapley,
        "sequences": np.asarray(sequences),
        "delta": game[7] - game[0],
    }

#!/usr/bin/env python3
"""Person-qualified joint evidence at a fixed whole-record token bag."""

from __future__ import annotations

import itertools

import numpy as np

from grain_research.construction import math_joint_diagnostic as J

POINTS = J.POINTS
FIELD_POINTS = np.asarray(list(itertools.product((-1, 1), repeat=5)), dtype=int)


def selected_pair_law(reference):
    """Enumerate independently sampled people, then retain complementary paths.

    Returns (t_A,t_B,b_A) mass after selection. Both class priors remain balanced,
    but the selected pair's classes are not independent.
    """
    counts = J.population_counts(reference).astype(float) / 100
    mass = np.empty((2, 2, 8))
    for a, b, i in itertools.product(range(2), range(2), range(8)):
        mass[a, b, i] = 0.25 * counts[a, i] * counts[b, 7 - i]
    selection_probability = float(mass.sum())
    return mass / selection_probability, selection_probability


def pair_posterior(choices, reference, person):
    """Condition on both complete observed paths under the explicit pair law."""
    if person not in (-1, 1):
        raise ValueError("Person -1=B or +1=A required")
    b = np.asarray(choices)
    if b.shape != (3,) or not np.isin(b, (-1, 1)).all():
        raise ValueError("Three signed person-A choices required")
    law, _ = selected_pair_law(reference)
    index = int(np.flatnonzero(np.all(POINTS == b, axis=1))[0])
    given = law[:, :, index]
    return float((given[0].sum() if person == 1 else given[:, 0].sum()) / given.sum())


def character_matrix():
    """Return all32 real characters, including the queried-person sign."""
    return np.stack(
        [
            np.prod(FIELD_POINTS[:, [i for i in range(5) if mask & (1 << i)]], axis=1)
            for mask in range(32)
        ],
        axis=1,
    ).astype(float)


def coefficients(field):
    """Retain all finite person/configuration/reference terms, not a fitted axis."""
    x = np.asarray(field, dtype=float)
    if x.shape[:5] != (2, 2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete five-factor field required")
    return np.tensordot(
        character_matrix().T / 32, x.reshape((32, *x.shape[5:])), axes=(1, 0)
    )


def reconstruct(beta):
    """Rebuild the complete field from every signed coefficient."""
    b = np.asarray(beta, dtype=float)
    if b.shape[0] != 32 or not np.isfinite(b).all():
        raise ValueError("All32 finite coefficients required")
    return np.tensordot(character_matrix(), b, axes=(1, 0)).reshape(
        (2, 2, 2, 2, 2, *b.shape[1:])
    )


def person_sectors(field):
    """Return common and signed A-minus-B readings; neither is pure psychology."""
    x = np.asarray(field, dtype=float)
    if x.shape != (2, 2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete scalar person field required")
    return (x[..., 1] + x[..., 0]) / 2, (x[..., 1] - x[..., 0]) / 2


def exchange_defect(field):
    """Quantify simultaneous global actor renaming and target renaming."""
    x = np.asarray(field, dtype=float)
    if x.shape != (2, 2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete scalar person field required")
    delta = x - np.flip(x, axis=(0, 1, 2, 4))
    return {
        "rms": float(np.sqrt(np.mean(delta**2))),
        "maximum": float(abs(delta).max()),
    }


def bag_observer(field):
    """Project to an arbitrary reference/person observer that cannot see binding.

    All eight assignment histories have the same bag. Averaging assignments is
    the exact conditional expectation on this finite design, not a learned
    predictor or an erasure operation on natural people.
    """
    x = np.asarray(field, dtype=float)
    if x.shape != (2, 2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete scalar person field required")
    return np.broadcast_to(x.mean(axis=(0, 1, 2), keepdims=True), x.shape).copy()

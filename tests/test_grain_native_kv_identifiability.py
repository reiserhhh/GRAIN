#!/usr/bin/env python3
"""Independent algebraic tests of the HM132 measurement-construction identities."""

import itertools

import numpy as np

from grain_research.construction import math_native_access as A
from grain_research.construction import math_native_kv_identity as M


def test_diagonal_restriction_has_eight_unidentifiable_directions():
    """Natural precision cannot separate all16 independently varying characters."""
    coefficients = np.eye(16)
    restriction = np.column_stack([M.restrict_coefficients(c) for c in coefficients])
    assert np.linalg.matrix_rank(restriction) == 8
    assert restriction.shape[1] - np.linalg.matrix_rank(restriction) == 8
    for left, right in (
        (0, 6),
        (1, 7),
        (2, 4),
        (3, 5),
        (8, 14),
        (9, 15),
        (10, 12),
        (11, 13),
    ):
        assert np.array_equal(restriction[:, left], restriction[:, right])


def test_qualified_author_law_retains_owner_content_organization():
    """Equal content/owner marginals can give different query-person scores."""
    rng = np.random.default_rng(132)
    cube = rng.normal(size=(2, 2, 2, 2, 6))
    eta = 0.4
    signs = (-1, 1)
    gamma, beta = M.characters(cube), A.characters(M.diagonal(cube))
    for p_index, p in enumerate(signs):
        contrast = np.zeros(6)
        margins = []
        for orientation in (-1, 1):
            law = np.empty((2, 2))
            for b_index, o_index in itertools.product(range(2), repeat=2):
                b, o = signs[b_index], signs[o_index]
                law[b_index, o_index] = (1 + orientation * eta * b * o) / 4
                contrast += (
                    orientation
                    * law[b_index, o_index]
                    * cube[b_index, o_index, o_index, p_index]
                )
            margins.append((law.sum(axis=0), law.sum(axis=1)))
        np.testing.assert_array_equal(margins[0], margins[1])
        np.testing.assert_allclose(
            contrast, 2 * eta * (beta[3] + p * beta[7]), atol=1e-14
        )
        np.testing.assert_allclose(
            contrast,
            2 * eta * (gamma[3] + gamma[5] + p * (gamma[11] + gamma[13])),
            atol=1e-14,
        )

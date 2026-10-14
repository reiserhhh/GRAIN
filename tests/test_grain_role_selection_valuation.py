#!/usr/bin/env python3
"""An exact counterexample separating key-code and readout interactions."""

import itertools

import numpy as np

from grain_research.construction import math_native_access as A
from grain_research.construction import math_native_key_sectors as M


def test_softmax_generic_role_selector_produces_behavior_owner_person_joint():
    """No joint key code is needed for a nonzero qualified behavioral read."""
    kappa, query, content = 0.7, 0.9, 0.4
    keys = np.zeros((2, 2, 2))
    cube = np.zeros((2, 2, 2))
    for bi, oi, pi in itertools.product(range(2), repeat=3):
        b, o, p = 2 * bi - 1, 2 * oi - 1, 2 * pi - 1
        k = np.array([o * kappa, -o * kappa])
        v = np.array([b * content, -b * content])
        logits = p * query * k
        weights = np.exp(logits - logits.max())
        cube[bi, oi, pi] = weights @ v / weights.sum()
        keys[bi, oi] = k
    assert np.array_equal(M.key_characters(keys)[3], np.zeros(2))
    beta = A.characters(cube)
    np.testing.assert_allclose(beta[7], content * np.tanh(query * kappa))
    np.testing.assert_allclose(beta[:7], np.zeros(7), atol=1e-16)


def test_paired_author_laws_retain_generic_selection_effect():
    """The same mechanism survives multi-evidence composition, not one seed."""
    beta = np.zeros(8)
    beta[7] = 0.4 * np.tanh(0.63)
    score = M.author_difference(beta, eta=0.5)
    assert score[0] < 0 < score[1]
    np.testing.assert_allclose(score, [-beta[7], beta[7]])

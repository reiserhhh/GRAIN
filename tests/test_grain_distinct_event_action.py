#!/usr/bin/env python3
"""Finite deletion laws and missingness, not synthetic personality validation."""

import numpy as np
import pytest

from grain_research.construction import math_distinct_event_action as M


def example():
    """Return an unbalanced pair panel with multiple distinct conditions."""
    rng = np.random.default_rng(190)
    x = rng.normal(size=(12, 3))
    c = np.array([0] * 7 + [1] * 5)
    w = rng.normal(size=(5, 3, 3))
    return x, c, w + w.transpose(0, 2, 1)


def test_distinct_pair_matrix_has_zero_diagonal_and_zero_translation():
    x, c, _ = example()
    b = M.pair_weights(c)
    np.testing.assert_array_equal(np.diag(b), 0)
    np.testing.assert_allclose(b.sum(0), 0, atol=1e-16)
    np.testing.assert_allclose(M.geometries(x + 2, c), M.geometries(x, c), atol=1e-14)


def test_all_signed_deletion_pieces_sum_to_unequal_event_action():
    x, c, w = example()
    a = M.deletion_action(x, c, np.array([0, 2, 8, 9]), w)
    np.testing.assert_allclose(a["pieces"].sum(0), a["delta"][1], atol=1e-14)
    np.testing.assert_allclose(a["delta"][0], a["delta"][1:].sum(0), atol=1e-14)
    assert np.any(a["pieces"] < 0)


def test_same_condition_counts_leave_choice_piece_zero_not_all_action():
    x, _, w = example()
    c = np.array([0] * 8 + [1] * 4)
    a = M.deletion_action(x, c, np.array([0, 1, 8]), w)
    np.testing.assert_allclose(a["pieces"][0], 0)
    assert np.linalg.norm(a["pieces"][1:]) > 0


def test_balanced_choice_zero_does_not_erase_pair_geometry():
    x, _, w = example()
    c = np.array([0] * 6 + [1] * 6)
    a = M.deletion_action(x, c, np.array([0, 1, 6, 7]), w)
    np.testing.assert_array_equal(a["delta"], 0)
    assert np.linalg.norm(M.geometries(x, c)[1]) > 0


def test_missing_or_singleton_condition_is_not_neutral_zero():
    x, c, w = example()
    assert M.deletion_action(x, c, np.array([7, 8, 9, 10]), w) is None
    z = M.author_mean(np.arange(8.0).reshape(2, 2, 2), np.array([[1, 0], [0, 0]]))
    np.testing.assert_array_equal(z[0], [0, 1])
    assert np.isnan(z[1]).all()
    assert M.contact(np.array([1, 1]), np.array([0, 2]))["r"] is None


def test_orthogonal_change_preserves_each_action_piece():
    x, c, w = example()
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    old = M.deletion_action(x, c, np.array([0, 2, 8, 9]), w)
    new = M.deletion_action(x @ q, c, np.array([0, 2, 8, 9]), q.T @ w @ q)
    for k in old:
        np.testing.assert_allclose(old[k], new[k], atol=1e-14)


def test_bad_event_addresses_cannot_silently_duplicate_evidence():
    x, c, w = example()
    for d in ([0, 0], [-1], [12], [0.5]):
        with pytest.raises(ValueError):
            M.deletion_action(x, c, np.array(d), w)

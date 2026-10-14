#!/usr/bin/env python3
"""Metadata-free scoring and finite author-law construction controls."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_relational_dual_frame as M


def rows(cells):
    """Declare only generation-side factorial metadata."""
    return [
        {"wording": w, "family": f, "condition": 2 * c - 1, "polarity": b, "owner": o}
        for w, (f, c), b, o in itertools.product(range(2), cells, (-1, 1), (-1, 1))
    ]


def test_dual_preserves_source_coordinates_and_complement():
    rng = np.random.default_rng(125)
    d = rng.normal(size=(17, 4))
    frame = M.fit_frame(rng.normal(size=(10, 17)), d)
    np.testing.assert_allclose(frame["dual"] @ d, np.eye(4), atol=1e-12)
    x = rng.normal(size=(8, 17))
    scored = M.score_vectors(x, frame)
    np.testing.assert_allclose(
        frame["center"] + scored["coordinates"] @ d.T + scored["complement"], x
    )
    np.testing.assert_allclose(scored["complement"] @ d, 0, atol=1e-12)


def test_rank_failure_is_not_zero_filled():
    with pytest.raises(ValueError, match="Rank-deficient"):
        M.fit_frame(np.ones((4, 3)), np.ones((3, 2)))


def test_identical_numeric_input_cannot_invent_role_difference():
    frame = M.fit_frame(np.eye(3), np.eye(3)[:, :2])
    scored = M.score_vectors(np.ones((2, 3)), frame)
    np.testing.assert_array_equal(scored["coordinates"][0], scored["coordinates"][1])


def test_profile_law_keeps_all_margins_and_lexical_distribution():
    cells = list(itertools.product(range(6), range(2)))
    bank = rows(cells)
    theta = np.stack((np.zeros((6, 2)), np.full((6, 2), 0.6)))
    p = M.make_profile_laws(bank, theta, cells)
    for field in ("wording", "family", "condition", "polarity", "owner"):
        for value in {r[field] for r in bank}:
            ids = [i for i, r in enumerate(bank) if r[field] == value]
            np.testing.assert_allclose(p[:, ids].sum(1), p[0, ids].sum())
    g = np.array([[r["family"], r["condition"], r["polarity"]] for r in bank])
    error, _ = M.lexical_law_distance(g, p)
    assert error < 1e-12


def test_absent_condition_parameter_cannot_change_observed_scores():
    cells = [(f, 0) for f in range(6)]
    bank = rows(cells)
    theta = np.zeros((2, 6, 2))
    theta[1, :, 1] = 0.6
    p = M.make_profile_laws(bank, theta, cells)
    np.testing.assert_array_equal(p[0], p[1])
    scored = M.compose_profiles(p, {"coordinates": np.arange(48)[:, None]})
    np.testing.assert_array_equal(scored["coordinates"][0], scored["coordinates"][1])


def test_author_contrast_identity_uses_vectors_not_application_labels():
    cells = list(itertools.product(range(6), range(2)))
    bank = rows(cells)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(96, 20))
    d = M.action_columns(x, bank, cells)
    frame = M.fit_frame(x, d)
    theta = rng.uniform(-0.6, 0.6, size=(6, 2))
    p = M.make_profile_laws(bank, np.stack((theta, -theta)), cells)
    scored = M.compose_profiles(p, M.score_vectors(x, frame))
    np.testing.assert_allclose(
        12 * np.diff(scored["coordinates"], axis=0)[0] / -2,
        theta.ravel(),
        atol=1e-11,
    )

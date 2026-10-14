#!/usr/bin/env python3
"""Synthetic finite-orbit checks independent of historical personality data."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_profile_reader as P
from grain_research.construction import math_source_anchor_exchange as M


def labels():
    """Two high/two low authors in every exact other-four stratum."""
    return np.repeat(np.array(list(itertools.product((0, 1), repeat=5))), 2, axis=0)


def test_old_matched_field_is_identical():
    y = labels()
    x = np.random.default_rng(11).normal(size=(len(y), 7))
    np.testing.assert_allclose(
        np.stack([M.statistic(x, y, q) for q in range(5)]),
        P.construct(x, y)["fields"][:, 1],
        atol=2e-16,
    )


def test_exact_enumerated_joint_moments():
    y = labels()
    b, _, w, _ = M.design(y, 0)
    x = np.zeros((len(y), 3))
    ix = np.flatnonzero(b == 0)
    x[ix] = [[1, 3, 0], [2, 8, 1], [6, 5, -2], [4, 2, 3]]
    orbit = []
    for high in itertools.combinations(range(4), 2):
        yh = y.copy()
        yh[ix, 0] = 0
        yh[ix[list(high)], 0] = 1
        orbit.append(M.statistic(x, yh, 0))
    orbit = np.array(orbit)
    mean, cov = M.exact_moments(x, y, 0, joint=True)
    np.testing.assert_allclose(orbit.mean(0), mean, atol=1e-16)
    np.testing.assert_allclose(orbit.T @ orbit / len(orbit), cov, atol=1e-16)
    np.testing.assert_allclose(M.exact_moments(x, y, 0)[1], cov.diagonal())
    assert w[0] > 0


def test_one_permutation_preserves_all_profiles():
    y = labels()
    original = np.bincount(y @ (2 ** np.arange(5)), minlength=32)
    for q in range(5):
        rows = M.assignment_indices(y, q, 7, 73)
        M.validate_assignments(y, q, rows)
        for row in rows:
            yh = y.copy()
            yh[:, q] = y[row, q]
            np.testing.assert_array_equal(np.delete(yh, q, 1), np.delete(y, q, 1))
            np.testing.assert_array_equal(
                np.bincount(yh @ (2 ** np.arange(5)), minlength=32), original
            )


def test_paired_views_not_independent():
    y = labels()
    x = np.random.default_rng(8).normal(size=(len(y), 1))
    twin = np.concatenate((x, 3 * x), axis=1)
    _, cov = M.exact_moments(twin, y, 3, joint=True)
    np.testing.assert_allclose(cov[0, 1], 3 * cov[0, 0])
    ref = M.references(twin, y, 3, M.assignment_indices(y, 3, 13, 55))
    np.testing.assert_allclose(ref[:, 1], 3 * ref[:, 0], atol=1e-15)


def test_projection_equals_contact():
    y = labels()
    rng = np.random.default_rng(33)
    x, actions, native = (
        rng.normal(size=(len(y), 9)),
        rng.normal(size=(11, 9)),
        rng.normal(size=11),
    )
    a = (actions - actions.mean(0)).T @ (native - native.mean()) / 10
    field = M.statistic(x, y, 2)
    direct = np.cov(actions @ field, native, ddof=1)[0, 1]
    np.testing.assert_allclose(M.statistic((x @ a)[:, None], y, 2)[0], direct)


def test_group_constants_are_exact_null():
    y = labels()
    b = M.design(y, 2)[0]
    values = np.stack((b, b**2), axis=1)
    np.testing.assert_allclose(M.statistic(values, y, 2), 0, atol=1e-15)
    np.testing.assert_array_equal(M.exact_moments(values, y, 2)[1], 0)
    assert M.rank_account(0, np.zeros(3), 0)["two_sided_rank"] is None


def test_reference_is_literal_label_change():
    y = labels()
    x = np.random.default_rng(9).normal(size=(len(y), 4))
    ix = M.assignment_indices(y, 1, 5, 12)
    ref = M.references(x, y, 1, ix)
    for i, row in enumerate(ix):
        yh = y.copy()
        yh[:, 1] = y[row, 1]
        np.testing.assert_allclose(ref[i], M.statistic(x, yh, 1), atol=2e-16)


def test_negative_contact_is_not_inverted():
    out = M.rank_account(-4, np.array([-1, 0, 1]), 1)
    assert out["standardized"] == -4
    assert out["upper_rank"] == 1
    assert out["two_sided_rank"] == 0.25


def test_holm_keeps_full_declared_family():
    np.testing.assert_allclose(
        M.holm([0.005, 0.02, 0.1, 1, 0.01]), [0.025, 0.06, 0.2, 1, 0.04]
    )
    assert M.holm([None, 0.01]) == [None, 0.02]


def test_invalid_assignments_rejected():
    y = labels()
    ix = M.assignment_indices(y, 0, 2, 4)
    ix[0, 0], ix[0, 2] = ix[0, 2], ix[0, 0]
    with pytest.raises(ValueError, match="within-stratum"):
        M.references(np.ones((len(y), 1)), y, 0, ix)


def test_missing_and_nonfinite_are_not_silent_zero():
    y = labels()
    y[:, 0] = 0
    with pytest.raises(ValueError, match="Unobserved"):
        M.design(y, 0)
    with pytest.raises(ValueError, match="Finite"):
        M.exact_moments(np.full((64, 1), np.nan), labels(), 0)
    with pytest.raises(ValueError, match="positive"):
        M.assignment_indices(labels(), 0, 0, 2)


def test_fixed_seed_and_no_gaussian_p_value():
    y = labels()
    np.testing.assert_array_equal(
        M.assignment_indices(y, 4, 3, 7), M.assignment_indices(y, 4, 3, 7)
    )
    out = M.rank_account(1, np.array([1, 2, -2]), 1)
    assert out["two_sided_rank"] == 1
    assert set(out) == {"orbit_sd", "standardized", "upper_rank", "two_sided_rank"}

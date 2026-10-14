#!/usr/bin/env python3
"""Finite paired-orbit and mechanism-account tests, without models or data."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_anchor_carrier_contrasts as C
from grain_research.construction import math_source_anchor_exchange as M


def labels():
    """Supply every32 binary profile twice, without missing strata."""
    return np.tile(np.array(list(itertools.product((0, 1), repeat=5))), (2, 1))


def test_signed_contrasts_and_telescope():
    rng = np.random.default_rng(213)
    a = rng.normal(size=(8, 2, 6, 5, 4))
    r = C.accounts(a)
    assert r.shape == (8, 2, 12, 3, 5, 4)
    np.testing.assert_allclose(r[:, :, 6, 0], a[:, :, 3] - a[:, :, 5])
    np.testing.assert_allclose(r[:, :, 7, 0], a[:, :, 0] - a[:, :, 1])
    np.testing.assert_allclose(r[..., 0, :, :], r[..., 1, :, :] + r[..., 2, :, :])
    assert C.telescoping_error(r) < 2e-15


def test_raw_self_control_is_not_specific_self_control():
    a = np.zeros((4, 2, 6, 5, 4))
    a[:, :, 3, 0] = 2
    r = C.accounts(a)
    np.testing.assert_array_equal(r[:, :, 6, 0, 3], 0)
    np.testing.assert_allclose(r[:, :, 6, 2, 3], -0.4)


def test_paired_orbit_contrast_not_independent_variance_sum():
    y, rng = labels(), np.random.default_rng(2)
    a = rng.normal(size=(64, 1))
    t = np.concatenate((a, 2 * a), axis=1)
    ix = M.assignment_indices(y, 2, 17, 213)
    covariance = M.exact_moments(t, y, 2, joint=True)[1]
    c = np.array([-1.0, 1.0])
    variance = M.exact_moments(t @ c[:, None], y, 2)[1][0]
    assert variance == pytest.approx(c @ covariance @ c)
    assert not np.isclose(variance, np.trace(covariance))
    np.testing.assert_allclose(
        M.references(t, y, 2, ix) @ c,
        M.references(t @ c[:, None], y, 2, ix)[:, 0],
    )


def test_exact_local_orbit_variance_by_enumeration():
    y = labels()
    b = M.design(y, 2)[0]
    local = np.flatnonzero(b == 0)
    t = np.zeros((64, 2))
    t[local] = [[1, 2], [3, -1], [0, 5], [2, 4]]
    permutations = np.tile(np.arange(64), (24, 1))
    permutations[:, local] = np.array(list(itertools.permutations(local)))
    refs = M.references(t, y, 2, permutations)
    mean, covariance = M.exact_moments(t, y, 2, joint=True)
    np.testing.assert_allclose(refs.mean(0), mean, atol=1e-16)
    np.testing.assert_allclose(np.cov(refs.T, ddof=0), covariance, atol=1e-16)


def test_identical_modes_have_undefined_not_successful_rank():
    y, a = labels(), np.ones((64, 1))
    variance = M.exact_moments(a - a, y, 2)[1][0]
    result = M.rank_account(0, np.zeros(199), variance)
    assert result["two_sided_rank"] is None


def test_projections_equal_direct_field_response_covariance():
    y, rng = labels(), np.random.default_rng(12)
    h, x = rng.normal(size=(64, 2, 7)), rng.normal(size=(8, 2, 7))
    r = C.accounts(rng.normal(size=(8, 2, 6, 5, 4)))
    t = C.projections(h, x, r)
    p = C.primary_columns(t, 2)
    fields = M.design(y, 2)[3] @ h.mean(1)
    expected = []
    for channel in (6, 7):
        a, b = x.mean(1) @ fields, r[:, :, channel, 2, 2, 2].mean(1)
        expected.append(np.cov(a, b, ddof=1)[0, 1])
    np.testing.assert_allclose(M.statistic(p, y, 2), expected, atol=1e-15)


@pytest.mark.parametrize("shape", [(8, 1, 6, 5, 4), (8, 2, 5, 5, 4), (1, 2, 6, 5, 4)])
def test_malformed_mode_support_rejected(shape):
    with pytest.raises(ValueError):
        C.accounts(np.zeros(shape))


def test_nonfinite_observation_rejected():
    a = np.zeros((4, 2, 6, 5, 4))
    a[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        C.accounts(a)


def test_wrong_vector_frame_rejected():
    r = C.accounts(np.zeros((4, 2, 6, 5, 4)))
    with pytest.raises(ValueError):
        C.projections(np.zeros((64, 2, 7)), np.zeros((4, 2, 8)), r)


def test_cross_stratum_assignments_rejected():
    y = labels()
    ix = np.arange(64)[None, :]
    ix[:, [0, 1]] = ix[:, [1, 0]]
    with pytest.raises(ValueError):
        M.references(np.zeros((64, 2)), y, 2, ix)

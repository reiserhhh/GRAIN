#!/usr/bin/env python3
"""Known-world dependence and ledger checks, including genuinely independent laws."""

import numpy as np
import pytest

from grain_research.construction import math_action_representation as M
from grain_research.construction.math_response_function import response_components


def test_nonlinear_dependence_exists_when_linear_covariance_is_zero():
    x = np.arange(-2.0, 3)[:, None]
    y = x**2
    assert abs(np.mean(x * (y - y.mean()))) < 1e-15
    result = M.distance_account(M.centered_distances(x), M.centered_distances(y))
    assert result["distance_covariance_squared"] > 0.3
    assert 0 < result["distance_correlation_squared"] < 1


def test_full_cartesian_law_has_zero_dependence_not_a_fake_null():
    x, y = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    result = M.distance_account(
        M.centered_distances(x.reshape(-1, 1)), M.centered_distances(y.reshape(-1, 1))
    )
    assert abs(result["distance_covariance_squared"]) < 1e-14


def test_geometry_is_rotation_translation_invariant_and_has_zero_centered_sums():
    rng = np.random.default_rng(169)
    x = rng.normal(size=(12, 7))
    q = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    a = M.centered_distances(x)
    np.testing.assert_allclose(a, M.centered_distances(x @ q + 10), atol=1e-13)
    np.testing.assert_allclose(a.sum(0), 0, atol=1e-13)


def test_pooled_condition_dependence_is_kept_separately_from_blocked_zero():
    x = np.array([[0.0], [0], [0], [5], [5], [5]])
    a, b, pieces = M.dependence_accounts(x, x, [0] * 3 + [1] * 3)
    assert M.distance_account(a, b)["distance_covariance_squared"] > 0
    assert M.stratified_account(pieces, 6)["distance_covariance_squared"] == 0


def test_whole_profile_permutation_keeps_block_and_all_rows():
    groups = np.array([0] * 3 + [1] * 4)
    first = M.profile_permutations(groups, 19, 169)
    np.testing.assert_array_equal(first, M.profile_permutations(groups, 19, 169))
    profiles = np.arange(28).reshape(7, 2, 2)
    for p in first:
        np.testing.assert_array_equal(groups[p], groups)
        np.testing.assert_array_equal(np.sort(p), np.arange(7))
        np.testing.assert_array_equal(profiles[p][:, 1, 1], profiles[p][:, 0, 0] + 3)


def test_distance_permutation_matches_fresh_calculation_and_retains_constant_null():
    rng = np.random.default_rng(13)
    x, y, groups = rng.normal(size=(9, 3)), rng.normal(size=(9, 4)), [0] * 3 + [1] * 6
    permutations = M.profile_permutations(groups, 7, 169)
    accounts = M.dependence_accounts(x, y, groups)
    null = M.permuted_dependence(accounts, permutations)
    for i, p in enumerate(permutations):
        a, b, pieces = M.dependence_accounts(x, y[p], groups)
        np.testing.assert_allclose(
            null[i, 0], M.distance_account(a, b)["distance_covariance_squared"]
        )
        np.testing.assert_allclose(
            null[i, 1], M.stratified_account(pieces, 9)["distance_covariance_squared"]
        )
    constant = M.dependence_accounts(x, np.ones((9, 2)), groups)
    np.testing.assert_array_equal(M.permuted_dependence(constant, permutations), 0)


def test_source_evidence_ledger_reconstructs_all_signed_accounts():
    rng = np.random.default_rng(41)
    x, y, d = (
        rng.normal(size=(12, 7)),
        rng.normal(size=(12, 2, 4, 2)),
        rng.normal(size=(8, 7)),
    )
    groups = [0] * 3 + [1] * 4 + [2] * 5
    _, c, w, b = response_components(x, y, groups)
    keys, weights, ledger = M.source_group_ledger(x, y, groups, d)
    np.testing.assert_array_equal(keys, [0, 1, 2])
    np.testing.assert_allclose(weights.sum(1), 0, atol=1e-14)
    np.testing.assert_allclose(ledger[:, 0], ledger[:, 1] + ledger[:, 2], atol=1e-14)
    target = np.einsum("pf,a...f->pa...", d, np.stack([c, w, b]))
    np.testing.assert_allclose(ledger.sum(2), target, atol=1e-14)
    assert (ledger < 0).any() and (ledger > 0).any()


def test_ledger_gauge_and_constant_response_are_not_author_signal():
    rng = np.random.default_rng(4)
    x, y, d = rng.normal(size=(8, 3)), rng.normal(size=(8, 2)), rng.normal(size=(5, 3))
    first = M.source_group_ledger(x, y, [0] * 4 + [1] * 4, d)
    other = M.source_group_ledger(x + 3, y + 5, [0] * 4 + [1] * 4, d)
    np.testing.assert_allclose(first[2], other[2], atol=1e-14)
    constant = M.source_group_ledger(x, np.ones((8, 2)), [0] * 4 + [1] * 4, d)
    np.testing.assert_array_equal(constant[2], 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_observations_are_rejected(bad):
    with pytest.raises(ValueError):
        M.centered_distances([[0], [bad]])
    with pytest.raises(ValueError):
        M.source_group_ledger([[0], [bad]], [[0], [1]], [0, 0], [[1], [2]])


def test_invalid_blocks_and_cross_block_permutations_are_rejected():
    with pytest.raises(ValueError):
        M.dependence_accounts([[0], [1]], [[0], [1]], [0, 1])
    accounts = M.dependence_accounts(
        np.arange(4)[:, None], np.arange(4)[:, None], [0, 0, 1, 1]
    )
    with pytest.raises(ValueError):
        M.permuted_dependence(accounts, [[2, 3, 0, 1]])

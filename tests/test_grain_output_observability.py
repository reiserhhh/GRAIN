#!/usr/bin/env python3
"""Output geometry, source-orbit means and distinct-author response bookkeeping."""

import copy
from itertools import permutations

import numpy as np
import pytest

from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_output_observability as M


def fixture():
    """Small positive vocabulary fixture, not psychological discovery data."""
    rng = np.random.default_rng(284)
    return rng.normal(size=(6, 9)), rng.dirichlet(np.ones(9))


def bank():
    """Complete ten-item balanced keys for five arithmetic query fixtures."""
    return [
        {"item": j, "target": j // 10, "pole": 1 if j % 10 < 4 else -1}
        for j in range(50)
    ]


def test_chain_rule_and_conditional_centering_match():
    z, p = fixture()
    parts = M.fisher_parts(z, p, [0, 1])
    np.testing.assert_allclose(parts[0], parts[1:].sum(0), atol=1e-14)
    centered = z - (z @ p)[:, None]
    np.testing.assert_allclose(parts[0], (centered * p) @ centered.T, atol=1e-14)
    for part in parts:
        assert np.linalg.eigvalsh(part).min() > -1e-12


def test_row_gauge_and_joint_hidden_basis_change_invariant():
    z, p = fixture()
    shifted = z + np.arange(6)[:, None]
    np.testing.assert_allclose(
        M.fisher_parts(z, p, [0, 1]), M.fisher_parts(shifted, p, [0, 1]), atol=1e-13
    )
    rng = np.random.default_rng(285)
    h, w = rng.normal(size=(6, 3)), rng.normal(size=(9, 3))
    a = np.array([[2, 0.5, 0], [0, 1, 0.2], [0.1, 0, 1]])
    np.testing.assert_allclose(h @ w.T, (h @ a.T) @ (w @ np.linalg.inv(a)).T)
    np.testing.assert_allclose(
        M.fisher_parts(h @ w.T, p, [0, 1]),
        M.fisher_parts((h @ a.T) @ (w @ np.linalg.inv(a)).T, p, [0, 1]),
        atol=1e-12,
    )


def test_binary_invisible_direction_can_remain_vocabulary_visible():
    p = np.array([0.3, 0.3, 0.2, 0.2])
    z = np.array([[0, 0, 1, -1]], float)
    parts = M.fisher_parts(z, p, [0, 1])
    assert parts[1, 0, 0] == 0
    assert parts[0, 0, 0] == pytest.approx(0.4)
    assert parts[3, 0, 0] == pytest.approx(0.4)


def test_complete_key_hidden_response_equals_fixed_readout():
    rng = np.random.default_rng(286)
    states = rng.normal(size=(48, 2, 4, 50, 3))
    v = rng.normal(size=3)
    fields = M.joint_fields(states, bank())
    keyed = np.einsum("uocj,qj->uocq", states @ v, M.key_matrix(bank()))
    expected = (keyed[:, :, 0] + keyed[:, :, 1] - keyed[:, :, 2] - keyed[:, :, 3]) / 2
    np.testing.assert_allclose(fields @ v, expected, atol=1e-13)
    np.testing.assert_allclose(M.key_matrix(bank()).sum(1), 0, atol=1e-15)


def test_distinct_author_account_matches_literal_signed_pairs():
    a, b = np.array([1, -2, 1]), np.array([-2, 1, 1])
    k = np.array([[1, 2, -1], [3, 4, 5], [-2, 1, 6]], float)
    output = M.cross_author_accounts(a, b, k)
    expected = (
        sum(a[i] * b[j] * k[i, j] for i in range(3) for j in range(3) if i != j) / 6
    )
    assert output["distinct_authors"] == pytest.approx(expected)
    assert output["all_pairs"] == pytest.approx(
        2 / 3 * expected + output["same_author"] / 3
    )


def test_quadratic_orbit_mean_is_not_automatically_zero():
    y, groups = np.array([0, 0, 1, 1]), np.zeros(4, int)
    a = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], float)
    b = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], float)
    k = np.array([[2, 1, 0], [3, 0, 2], [1, -2, 1]], float)
    values = []
    for ix in permutations(range(4)):
        e = y[list(ix)] - 0.5
        values.append(
            M.cross_author_accounts(
                M.centered_scores(a @ e), M.centered_scores(b @ e), k
            )["distinct_authors"]
        )
    exact = M.orbit_mean(a, b, C.orbit_covariance(y, groups), k)
    assert exact == pytest.approx(np.mean(values), abs=1e-14)
    assert abs(exact) > 0.001


@pytest.mark.parametrize(
    "bad",
    [np.array([0.5, 0.5, 0]), np.array([0.2, 0.2, 0.2]), np.array([0.5, -0.1, 0.6])],
)
def test_invalid_probability_reference_rejected(bad):
    with pytest.raises(ValueError):
        M.fisher_parts(np.ones((2, 3)), bad, [0, 1])


def test_constant_output_field_has_no_geometry_or_response_signal():
    p = np.array([0.3, 0.2, 0.5])
    z = np.array([[2, 2, 2], [-1, -1, -1]], float)
    np.testing.assert_allclose(M.fisher_parts(z, p, [0, 1]), 0, atol=1e-15)


def test_cold_accounts_reject_labels_and_duplicate_events():
    r = {
        "version": "HM284-v1",
        "author_keys": ["a", "b"],
        "left_event_ids": [["a0", "a1"], ["b0", "b1"]],
        "right_event_ids": [["a2", "a3"], ["b2", "b3"]],
        "source_readings": np.arange(20).reshape(2, 2, 5).tolist(),
        "cross_output_kernels": np.zeros((4, 5, 2, 2)).tolist(),
    }
    assert np.asarray(M.cold(r)["distinct_author_source_query_channels"]).shape == (
        5,
        5,
        4,
    )
    for key in ("labels", "duplicate"):
        bad = copy.deepcopy(r)
        if key == "labels":
            bad["big5"] = [1] * 5
        else:
            bad["right_event_ids"][0][0] = "a0"
        with pytest.raises(ValueError):
            M.cold(bad)

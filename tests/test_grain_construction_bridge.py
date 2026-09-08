#!/usr/bin/env python3
"""Synthetic tests for frozen transport and cross-query natural-response accounting."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.construction_bridge import (
    author_split,
    bootstrap_quality,
    candidate_predictions,
    covariance_account,
    covariance_matrix,
    crossed_predictions,
    fit_reference,
    quality,
    response_account,
    two_order_allocation,
    validate_arrays,
)


def fixture():
    """Small fully crossed data; one gradient vector for each author/view/query."""
    rng = np.random.default_rng(41)
    g, d = rng.normal(size=(8, 2, 3, 5)), rng.normal(size=(8, 2, 2, 3, 5))
    y, m = rng.normal(size=d.shape[:-1]), rng.normal(size=d.shape[:-1])
    return g, d, y, m


def test_author_hash_split_is_input_order_invariant():
    ids = [f"u{i}" for i in range(24)]
    mask = author_split(ids, "frozen")
    reverse = author_split(ids[::-1], "frozen")
    assert {x for x, keep in zip(ids, mask) if keep} == {
        x for x, keep in zip(ids[::-1], reverse) if keep
    }
    assert mask.sum() == 12
    with pytest.raises(ValueError):
        author_split(["same"] * 24, "bad")


def test_reference_does_not_use_evaluation_gradients_or_targets():
    g, d, y, m = fixture()
    validate_arrays(g, d, y, m)
    ref = np.arange(8) < 4
    first = fit_reference(g, y, ref)
    g[~ref] *= 999
    y[~ref] += 888
    second = fit_reference(g, y, ref)
    for x, z in zip(first, second):
        np.testing.assert_array_equal(x, z)


def test_crossed_axes_and_query_units():
    g = np.eye(3)
    d = np.arange(18).reshape(1, 1, 2, 3, 3)
    b = crossed_predictions(g, d)
    np.testing.assert_array_equal(b[0, 0, 1, 2], d[0, 0, 1, :, 2])
    means, scales = np.zeros((1, 2, 3)), np.array([1, 2, 3])
    models, bn = candidate_predictions(g, d, means, scales)
    np.testing.assert_allclose(bn, b / scales[None, None, None, :, None])
    np.testing.assert_allclose(
        models["private_evidence_private_reader"], np.diagonal(bn, axis1=-2, axis2=-1)
    )


def test_two_order_allocation_does_not_add_interaction_twice():
    allocation = two_order_allocation(np.array([[10.0, 14.0], [15.0, 21.0]]), 0, 1)
    assert allocation == {
        "difference": -11.0,
        "readout": -6.0,
        "evidence": -5.0,
        "interaction": 2.0,
    }


def test_four_response_terms_are_exact():
    g, d, y, m = fixture()
    local = np.einsum("avqd,aveqd->aveq", g, d)
    a = np.einsum("qd,aveqd->aveq", g.mean(axis=(0, 1)), d)
    np.testing.assert_allclose(sum(response_account(y, m, local, a).values()), y)


@pytest.mark.parametrize("centered", [True, False])
def test_four_covariance_terms_are_exact_and_crossed(centered):
    t = np.array([-1.0, 0.0, 1.0])[:, None, None]
    a = np.concatenate((2 * t, -3 * t), axis=-1)
    y = np.concatenate((3 * t, -5 * t), axis=-1)
    account = covariance_account(y, a, centered)
    np.testing.assert_allclose(
        sum(v for k, v in account.items() if k != "observed"), account["observed"]
    )
    assert account["observed"][0, 1] == -10
    assert account["functional"][0, 1] == -4


def test_cell_centering_removes_edit_effect_not_author_structure():
    t = np.arange(6)[:, None, None]
    cells = np.array([0.0, 100.0])[None, :, None]
    x = np.concatenate((t + cells, -t + cells), axis=-1)
    assert covariance_matrix(x)[0, 1] < 0
    assert covariance_matrix(x, centered_within_cell=False)[0, 1] > 0


def test_zero_outcomes_never_divide_by_casewise_response():
    stats = quality(np.zeros((4, 2)), np.ones((4, 2)))
    assert stats["r"] is None and stats["mse_over_zero"] is None
    boot = bootstrap_quality(np.zeros((4, 2)), np.ones((4, 2)), draws=10)
    assert boot["r_lower"] is None and boot["mse"] == 1
    g, _d, y, _m = fixture()
    with pytest.raises(ValueError):
        fit_reference(g, np.zeros_like(y), np.arange(8) < 4)


def test_shape_and_nonfinite_fail_closed():
    g, d, y, m = fixture()
    with pytest.raises(ValueError):
        validate_arrays(g, d, y[:, :, :1], m)
    d[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_arrays(g, d, y, m)

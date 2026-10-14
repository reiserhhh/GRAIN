#!/usr/bin/env python3
"""Known-answer checks for composition, support and reader-error accounting."""

import numpy as np
import pytest

from grain_research.response_accounting import (
    apply_operator,
    chronological_halves,
    comment_average,
    difference_operator,
    direct_difference,
    paired_energy_interval,
)


def test_pure_composition_change_is_not_response():
    c = np.array(["a", "a", "b", "a", "b", "b"])
    h = np.repeat([0, 1], 3)
    y = np.where(c == "a", 0.0, 3.0)[:, None]
    a, meta = difference_operator(c, h, np.ones(6, bool))
    result = apply_operator(a, y)
    np.testing.assert_allclose(result[:, 0], [1, 0, 1, 0], atol=1e-12)
    assert meta["shared_conditions"] == 2


def test_pure_within_change():
    c = ["a", "b", "a", "b"]
    h = [0, 0, 1, 1]
    y = np.array([0.0, 4.0, 2.0, 6.0])[:, None]
    a, _ = difference_operator(c, h, np.ones(4, bool))
    np.testing.assert_allclose(apply_operator(a, y)[:, 0], [2, 2, 0, 0])


def test_unshared_support_not_imputed():
    a, meta = difference_operator(["a", "a", "b", "b"], [0, 0, 1, 1], np.ones(4, bool))
    assert not meta["within_measured"]
    np.testing.assert_allclose(
        apply_operator(a, np.arange(4.0)[:, None])[:, 0], [2, 0, 0, 2]
    )


def test_joint_terms_reverse_and_independent_cell_check():
    rng = np.random.default_rng(9)
    c = np.array(["a", "a", "b", "d", "a", "b", "c", "c"])
    h = np.repeat([0, 1], 4)
    y = rng.normal(size=(8, 7))
    a, _ = difference_operator(c, h, np.ones(8, bool))
    reversed_a, _ = difference_operator(c, 1 - h, np.ones(8, bool))
    np.testing.assert_allclose(a, -reversed_a)
    r = apply_operator(a, y)
    np.testing.assert_allclose(r, direct_difference(c, h, y), atol=1e-14)
    np.testing.assert_allclose(r[0], r[1:].sum(0), atol=1e-14)


def test_origin_dependency_and_preserved_sum():
    c, h = ["a", "b", "a", "a"], [0, 0, 1, 1]
    y = np.arange(4.0)[:, None]
    a, _ = difference_operator(c, h, np.ones(4, bool))
    before, after = apply_operator(a, y), apply_operator(a, y + 10)
    np.testing.assert_allclose(before[:2], after[:2])
    assert before[2, 0] != after[2, 0]
    np.testing.assert_allclose(before[2:].sum(0), after[2:].sum(0))


def test_reader_error_propagates_exactly():
    rng = np.random.default_rng(1)
    c, h = ["a", "a", "b", "a", "c", "c"], [0, 0, 0, 1, 1, 1]
    a, _ = difference_operator(c, h, np.ones(6, bool))
    t, error = rng.normal(size=(6, 10)), rng.normal(size=(6, 10))
    np.testing.assert_allclose(
        apply_operator(a, t + error) - apply_operator(a, t),
        apply_operator(a, error),
        atol=1e-14,
    )


def test_missing_comments_are_not_zero_and_sentences_equalize():
    means, count = comment_average(
        np.array([[1.0], [3.0], [10.0]]), np.array([0, 0, 2]), 4
    )
    np.testing.assert_allclose(means[[0, 2], 0], [2, 10])
    assert np.isnan(means[[1, 3]]).all()
    a, _ = difference_operator(["a"] * 4, [0, 0, 1, 1], count > 0)
    assert apply_operator(a, means)[0, 0] == 8
    with pytest.raises(ValueError, match="missing"):
        apply_operator(np.ones((4, 4)), means)
    with pytest.raises(ValueError, match="Both halves"):
        difference_operator(
            ["a"] * 4, [0, 0, 1, 1], np.array([True, False, False, False])
        )


def test_time_order_and_tied_boundary():
    h, meta = chronological_halves([30.0, 10.0, 40.0, 20.0], ["a", "b", "c", "d"])
    np.testing.assert_array_equal(h, [1, 0, 1, 0])
    assert meta["strict_time_boundary"]
    _, meta = chronological_halves([10, 20, 20, 30], list("abcd"))
    assert not meta["strict_time_boundary"]
    with pytest.raises(ValueError):
        chronological_halves([1, np.nan], ["a", "b"])


def test_bootstrap_and_input_errors():
    d = paired_energy_interval(np.ones(10), np.zeros(10), draws=20)
    assert d["delta_mean_energy"] == d["ci_low"] == d["ci_high"] == 1
    with pytest.raises(ValueError):
        difference_operator(["a", "a"], [0, 2], np.ones(2, bool))
    with pytest.raises(ValueError):
        comment_average(np.ones((2, 1)), np.array([-1, 0]), 2)


def test_better_pointwise_error_can_give_worse_change_error():
    # A constant reader bias cancels in a contrast; a smaller directional bias does not.
    a, _ = difference_operator(["a"] * 4, [0, 0, 1, 1], np.ones(4, bool))
    constant = np.ones((4, 1))
    directional = np.array([-0.2, -0.2, 0.2, 0.2])[:, None]
    assert np.mean(directional**2) < np.mean(constant**2)
    assert abs(apply_operator(a, directional)[1, 0]) > abs(
        apply_operator(a, constant)[1, 0]
    )


def test_transformed_error_risk_needs_off_diagonal_covariance():
    a = np.array([-0.5, -0.5, 0.5, 0.5])
    # The same diagonal variances give different contrast risk under common errors.
    assert a @ np.eye(4) @ a == 1
    assert a @ np.ones((4, 4)) @ a == 0

#!/usr/bin/env python3
"""Known-null, hidden-response and cold observation construction controls."""

import numpy as np
import pytest

from grain_research.construction import math_author_response_observability as M


def test_equal_scores_can_hide_actual_intervention_response():
    n = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    d = np.array([[1], [-1], [-1], [1]], float)
    b = np.column_stack((np.ones(4), n))
    visible, hidden, _ = M.response_account(b, d)
    laws, active = M.positive_witnesses(hidden)
    assert active.all() and np.max(abs(visible)) < 1e-12
    assert np.all(laws > 0) and np.allclose(laws.sum(1), 1)
    contrast = laws[0, :, 0] - laws[1, :, 0]
    assert np.max(abs(contrast @ b)) < 1e-12
    assert contrast @ d[:, 0] > 0
    _, rest, _ = M.response_account(np.column_stack((b, d)), d)
    assert np.max(abs(rest)) < 1e-12


def test_observed_response_null_is_not_forced_to_be_new_structure():
    b = np.column_stack((np.ones(6), np.arange(6)))
    d = b @ np.array([[0.3, 1], [-0.2, 0.5]])
    _, hidden, _ = M.response_account(b, d)
    laws, active = M.positive_witnesses(hidden)
    assert not active.any()
    assert np.array_equal(laws[0], laws[1])


def test_group_masses_preserved_and_source_recoding_invariant():
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(8), 2)
    n, g, a, d = (rng.normal(size=(16, k)) for k in (3, 2, 3, 2))
    b = M.source_observers(n, g, a, groups)["native"]
    _, hidden, _ = M.response_account(b, d)
    laws, _ = M.positive_witnesses(hidden)
    assert np.max(abs((laws[0] - laws[1]).T @ b)) < 1e-12
    _, other, _ = M.response_account(b * np.arange(1, b.shape[1] + 1), d)
    assert np.max(abs(other - hidden)) < 1e-12
    choices = M.choice_columns(groups)
    assert np.allclose(choices.sum(1), 1)


def test_cold_stream_equals_explicit_event_sum_and_missing_condition_is_nan():
    x = np.arange(12).reshape(4, 3).astype(float)
    g = np.array([0, 0, 1, 1])
    ids = np.array([[0, 1, 0], [2, 3, 2]])
    out = M.cold_scores(x, g, ids)
    assert np.array_equal(out["mean"], x[ids].mean(1))
    assert np.isnan(out["conditional_mean"][0, 1]).all()
    order = M.cold_scores(x, g, ids[:, ::-1])
    assert np.array_equal(order["mean"], out["mean"])
    assert np.array_equal(out["event_counts"].sum(1), np.full(2, 3))


def test_unsupported_and_nonfinite_inputs_fail_without_false_zero():
    with pytest.raises(ValueError):
        M.response_account(np.ones((3, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError):
        M.positive_witnesses(np.ones((2, 1)), 1)
    with pytest.raises(ValueError):
        M.cold_scores(np.ones((2, 3)), [0, 0], np.array([[2]]))
    with pytest.raises(ValueError):
        M.observation_space(np.full((2, 2), np.nan))

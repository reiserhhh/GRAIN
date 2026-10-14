#!/usr/bin/env python3
"""Known-world checks for observed response construction, not persona prediction."""

import numpy as np
import pytest

from grain_research.construction import math_response_function as M


def test_total_decomposition_keeps_pure_condition_signal():
    x = np.array([[0.0, 1], [0, -1], [4, 1], [4, -1]])
    y = np.array([[0.0], [0], [2], [2]])
    origin, total, within, between = M.response_components(x, y, [0, 0, 1, 1])
    np.testing.assert_allclose(origin, [2, 0])
    np.testing.assert_allclose(total, within + between)
    np.testing.assert_allclose(within, 0)
    np.testing.assert_allclose(total, [[2, 0]])


def test_within_and_between_keep_signed_cancellation():
    x = np.array([[0.0], [2], [2], [4]])
    y = np.array([[0.0], [-2], [4], [2]])
    _, total, within, between = M.response_components(x, y, [0, 0, 1, 1])
    np.testing.assert_allclose(total, within + between)
    assert within.item() < 0 < between.item()


def test_cold_response_is_not_a_probability_or_conditional_mean():
    x = np.array([[-2.0], [2]])
    y = np.array([[0.0], [1]])
    m, c, _, _ = M.response_components(x, y, [0, 0])
    out = M.cold_readings([[10]], m, c)
    assert out.item() == 10


def test_numeric_cold_reading_affine_reference_and_order_identity():
    rng = np.random.default_rng(166)
    x, y = rng.normal(size=(12, 7)), rng.normal(size=(12, 2, 4, 2))
    m, c, w, b = M.response_components(x, y, [0] * 6 + [1] * 6)
    r = M.cold_readings(x, m, np.stack([c, w, b]))
    np.testing.assert_allclose(r[:, 0], r[:, 1] + r[:, 2], atol=1e-14)
    evenodd = M.order_fields(r)
    np.testing.assert_allclose(evenodd[..., 0] + evenodd[..., 1], r[..., 0])
    np.testing.assert_allclose(evenodd[..., 0] - evenodd[..., 1], r[..., 1])
    np.testing.assert_allclose(M.cold_readings(m[None], m, c), 0)


def test_relink_keeps_whole_profiles_and_each_condition_multiset():
    y = np.arange(72).reshape(12, 3, 2)
    groups = [0] * 6 + [1] * 6
    for shift in range(1, 6):
        z = M.relink_profiles(y, groups, shift)
        assert not np.array_equal(y, z)
        for ids in (slice(0, 6), slice(6, 12)):
            np.testing.assert_array_equal(
                np.sort(y[ids], axis=0), np.sort(z[ids], axis=0)
            )


def test_constant_response_gives_zero_function_not_fake_success():
    x = np.arange(12).reshape(6, 2)
    m, c, _, _ = M.response_components(x, np.ones((6, 4, 2)), [0] * 6)
    np.testing.assert_array_equal(M.cold_readings(x, m, c), 0)
    assert M.signed_alignment(np.zeros(6), np.arange(6))["cosine"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_source_and_cold_rejected(bad):
    with pytest.raises(ValueError):
        M.response_components([[0], [bad]], [[0], [1]], [0, 0])
    with pytest.raises(ValueError):
        M.cold_readings([[1]], [0], [[bad]])


def test_invalid_groups_order_and_relink_rejected():
    with pytest.raises(ValueError):
        M.response_components([[0], [1]], [[0], [1]], [0])
    with pytest.raises(ValueError):
        M.order_fields([0, 1, 2])
    with pytest.raises(ValueError):
        M.relink_profiles([[0], [1]], [0, 0], 2)

#!/usr/bin/env python3
"""Synthetic checks for the fixed companion observation contract."""

import numpy as np
import pytest

from grain_research.construction import math_companion_observation as M


def test_additive_corners_have_no_mixed_action():
    a = np.array([2.0, 0.5])
    b = np.array([-3.0, 7.0])
    assert M.mixed(b[:, None] + a[None, :]) == 0


def test_bilinear_four_corner_is_exact():
    rng = np.random.default_rng(243)
    a, b, t = rng.normal(size=(2, 7)), rng.normal(size=(2, 7)), rng.normal(size=(7, 7))
    table = np.array([[x @ t @ y for x in a] for y in b])
    assert M.mixed(table) == pytest.approx((a[0] - a[1]) @ t @ (b[0] - b[1]))


def test_reader_can_interact_with_additive_hidden_states():
    h00 = np.array([0.1, 0.3])
    a, b = np.array([0.9, -0.4]), np.array([-0.2, 0.7])
    h = np.array([[h00 + a + b, h00 + b], [h00 + a, h00]])
    rng = np.random.default_rng(24)
    result = M.head_accounts(h, rng.normal(size=(10, 2)), np.zeros(10))
    np.testing.assert_allclose(result["mixed_hidden"], 0, atol=1e-15)
    assert abs(result["reader"]) > 1e-4
    assert result["representation"] == pytest.approx(0, abs=1e-15)
    assert result["closure"] == pytest.approx(0, abs=1e-15)


def test_representation_and_reader_accounts_keep_cross_term():
    rng = np.random.default_rng(25)
    h = rng.normal(size=(4, 2, 2, 7))
    result = M.head_accounts(h, rng.normal(size=(10, 7)), rng.normal(size=10))
    np.testing.assert_allclose(
        result["total"], result["reader"] + result["representation"], atol=1e-13
    )
    assert np.any(result["reader"] * result["representation"] < 0)


def test_common_logit_shift_cannot_create_target_interaction():
    rng = np.random.default_rng(26)
    z = rng.normal(size=(4, 2, 2, 10))
    shift = rng.normal(size=(4, 2, 2, 1))
    for key in ("score", "log_odds"):
        np.testing.assert_allclose(
            M.read_logits(z)[key], M.read_logits(z + shift)[key], atol=1e-14
        )


def test_missing_variation_is_not_a_zero_contact():
    assert M.cosine(np.ones(5), np.arange(5)) is None
    with pytest.raises(ValueError):
        M.mixed(np.zeros((2, 3)))


def test_all_cross_targets_retained():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(24, 2, 5))
    result = np.asarray(M.contacts(x, x))
    assert result.shape == (5, 5)
    np.testing.assert_allclose(np.diag(result), 1)

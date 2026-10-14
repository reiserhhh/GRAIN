#!/usr/bin/env python3
"""Synthetic link-only and pre-link counterexamples for the span diagnostic."""

import numpy as np
import pytest

from scripts.diagnose_grain_math_span_readout import (
    gradient_integral,
    split_four_logits,
)


def test_additive_logits_can_have_nonlinear_score_interaction():
    z0 = np.array([0.0, -0.5, -1.0])
    p, c = np.array([1.0, 0.0, -0.5]), np.array([0.0, 0.7, -0.4])
    result = split_four_logits([z0, z0 + p, z0 + c, z0 + p + c], [0.0, 0.5, 1.0])
    assert result["pre_link_norm"] < 1e-14
    assert abs(result["pre_link_action"]) < 1e-14
    assert abs(result["link_curvature"]) > 1e-3
    assert result["link_integral_error"] < 1e-14


def test_pre_link_only_case_is_detected_without_link_interaction():
    z0 = np.array([0.0, -0.5, -1.0])
    full = z0 + [0.5, 0.0, -0.2]
    result = split_four_logits([z0, z0, z0, full], [0.0, 0.5, 1.0])
    assert abs(result["link_curvature"]) < 1e-14
    assert abs(result["pre_link_action"]) > 0.01
    assert result["pre_link_kl"] > 0
    assert result["kl_bregman_error"] < 1e-14


def test_four_corner_gauge_changes_do_not_create_interaction():
    rng = np.random.default_rng(72)
    z = rng.normal(size=(4, 10))
    values = np.linspace(0, 1, 10)
    expected = split_four_logits(z, values)
    actual = split_four_logits(z + np.array([5, -2, 11, 7])[:, None], values)
    for name in (
        "mixed_score",
        "pre_link_action",
        "link_curvature",
        "pre_link_norm",
        "pre_link_kl",
    ):
        assert actual[name] == pytest.approx(expected[name], abs=1e-14)


def test_pre_link_nonadditivity_can_be_invisible_to_one_score():
    # Symmetric extremes cancel in an ordinal mean, but not in its distribution.
    result = split_four_logits([[0, 0, 0]] * 3 + [[1, -1, 1]], [0, 0.5, 1])
    assert abs(result["mixed_score"]) < 1e-14
    assert result["pre_link_norm"] > 1
    assert result["pre_link_kl"] > 0.1


def test_path_integral_is_score_change_including_constant_gauge():
    z = np.array([0.0, 1.0, -1.0])
    d = np.array([0.3, -0.7, 0.2])
    values = np.array([0.0, 0.5, 1.0])
    p = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
    expected = (p(z + d) - p(z)) @ values
    assert gradient_integral(z, d, values) == pytest.approx(expected, abs=1e-14)
    assert gradient_integral(z, d + 4, values) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("shape", [(3, 10), (4, 3, 2)])
def test_incomplete_rectangle_is_rejected(shape):
    with pytest.raises(ValueError, match="four-corner"):
        split_four_logits(np.zeros(shape), np.arange(10))


def test_nonfinite_inputs_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        split_four_logits([[0, 0], [0, 0], [0, 0], [np.nan, 0]], [0, 1])

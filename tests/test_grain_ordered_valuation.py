#!/usr/bin/env python3
"""Finite worlds distinguish universal order from selected scalar agreement."""

import numpy as np
import pytest

from grain_research.construction import math_ordered_valuation as O


@pytest.mark.parametrize("sign", [1, -1])
def test_universal_order(sign):
    m = sign * np.array([-0.3, 0.1, 0.1, 0.1])
    c = O.certificate(m)
    assert c["status"] == (
        "ORDINALLY_NONNEGATIVE" if sign > 0 else "ORDINALLY_NONPOSITIVE"
    )
    assert np.all(sign * O.tail_actions(m) > 0)


def test_readout_dependent_world():
    m = np.array([0.2, -0.5, 0.3])
    c = O.certificate(m)
    assert c["status"] == "SCALE_DEPENDENT"
    assert O.readout(m, [0, 0.99, 1]) < 0
    assert O.readout(m, [0, 0.01, 1]) > 0


def test_all_strict_scales_inside_exact_bounds():
    rng = np.random.default_rng(179)
    m = np.array([0.2, -0.5, 0.2, 0.1])
    c = O.certificate(m)
    for _ in range(60):
        w = rng.dirichlet(np.ones(3))
        v = np.r_[0, np.cumsum(w)]
        v[-1] = 1
        value = O.readout(m, v)
        assert c["lower"] < value < c["upper"]
        assert value == pytest.approx(w @ O.tail_actions(m), abs=2e-15)


def test_generator_extrema_are_exact_not_search():
    m = np.array([0.2, -0.5, 0.2, 0.1])
    direct = [sum(m[j] for j in range(k, 4)) for k in range(1, 4)]
    assert np.allclose(O.tail_actions(m), direct, rtol=0, atol=1e-15)
    assert O.certificate(m)["lower"] == pytest.approx(min(direct), abs=2e-15)
    assert O.certificate(m)["upper"] == pytest.approx(max(direct), abs=2e-15)


def test_source_zero_is_not_certified_agreement():
    b, c = O.source_alignment(np.zeros(2), np.array([[-1, 1], [1, -1]]))
    assert b is None and c["status"] == "UNRESOLVED_SOURCE_ZERO"
    assert O.oriented_individual(0, [-1, 1])["status"] == "UNRESOLVED_SOURCE_ZERO"


def test_native_zero_is_not_positive():
    assert O.certificate(np.zeros(10))["status"] == "NUMERICALLY_UNRESOLVED_ZERO"


def test_universal_inverse_individual():
    c = O.oriented_individual(-0.2, [-0.3, 0.1, 0.1, 0.1])
    assert c["all_generators_strictly_negative"]


def test_alignment_keeps_author_aggregation():
    x = np.array([0.2, -0.1])
    m = np.array([[-0.2, 0.1, 0.1], [0.1, -0.05, -0.05]])
    b, _ = O.source_alignment(x, m)
    assert np.allclose(b, sum(x[u] * m[u] for u in range(2)) / np.linalg.norm(x))
    assert O.readout(b, [0, 0.5, 1]) == pytest.approx(
        x @ (m @ [0, 0.5, 1]) / np.linalg.norm(x)
    )


@pytest.mark.parametrize("bad", [[1, 0, 0], [np.nan, 0, 0], [1]])
def test_invalid_measures_rejected(bad):
    with pytest.raises(ValueError):
        O.certificate(bad)


@pytest.mark.parametrize("bad", [[0, 0.8, 0.2, 1], [0.1, 0.3, 0.6, 1], [0, 1]])
def test_invalid_scale_rejected(bad):
    with pytest.raises(ValueError):
        O.readout([-0.3, 0.1, 0.1, 0.1], bad)


def test_native_gauge_and_pair_distribution_account():
    x = np.random.default_rng(179).normal(size=(3, 2, 5, 10))
    p, m, plus, minus = O.native_action(x)
    _, shifted, _, _ = O.native_action(x + 17)
    assert np.allclose(m, shifted, rtol=0, atol=2e-15)
    assert np.allclose(m, 2 * (plus - minus), rtol=0, atol=2e-15)
    assert np.allclose(plus.sum(-1), 1)
    assert np.allclose(minus.sum(-1), 1)
    assert np.allclose(
        O.readout(m, np.arange(10) / 9),
        (p @ (np.arange(10) / 9))[..., 2]
        - (p @ (np.arange(10) / 9))[..., 1]
        - (p @ (np.arange(10) / 9))[..., 3]
        + (p @ (np.arange(10) / 9))[..., 4],
    )


def test_nonfinite_native_rejected():
    with pytest.raises(ValueError):
        O.native_action(np.full((5, 10), np.nan))

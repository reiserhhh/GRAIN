#!/usr/bin/env python3
"""Mechanism controls for exact hierarchy, not predictor accuracy tests."""

import numpy as np
import pytest

from grain_research.construction import math_hierarchical_query as M


def sample():
    """Return a nonuniform base with unequal nonempty event sizes."""
    return np.array([[0.3, -0.4, 1.0, -0.2, 0.7]]), np.array([0, 0, 1, 1, 1])


def test_between_only():
    base, ids = sample()
    f = M.factorize(base, np.array([[2, 2, -1, -1, -1]]), ids)
    assert np.allclose(f["within_only"], f["cold"])
    assert np.allclose(f["event_only"], f["native"])
    assert np.allclose(f["within_kl"], 0)


def test_normalized_within_only():
    base, ids = sample()
    first = M.factorize(base, np.array([[1, -1, 0.2, -0.3, 0.4]]), ids)
    f = M.factorize(base, first["within_log_tilt"], ids)
    assert np.allclose(f["mass0"], f["mass1"])
    assert np.allclose(f["event_only"], f["cold"])


def test_mixed_chain_and_constraints():
    base, ids = sample()
    f = M.factorize(base, np.array([[1, -1, 0.2, -0.3, 0.4]]), ids)
    assert np.allclose(f["total_kl"], f["between_kl"] + f["within_kl"])
    assert np.allclose(
        M.normalize(base + f["event_log_tilt"][:, ids] + f["within_log_tilt"]),
        f["native"],
    )
    for g in (0, 1):
        ix = ids == g
        assert np.allclose(np.exp(f["within_only"][:, ix]).sum(1), f["mass0"][:, g])
        a, b = np.exp(f["event_only"][:, ix]), np.exp(f["cold"][:, ix])
        assert np.allclose(a / a.sum(1)[:, None], b / b.sum(1)[:, None])


def test_naive_zero_mean_can_move_event_mass():
    f = M.factorize(np.zeros((1, 4)), np.array([[2, -2, 0, 0]]), np.array([0, 0, 1, 1]))
    assert f["naive_mass_drift"][0] > 0.2
    assert np.allclose(np.exp(f["within_only"][:, :2]).sum(1), 0.5)


def test_gauge_invariance_and_zero():
    base, ids = sample()
    f = M.factorize(base, np.zeros_like(base), ids)
    shifted = M.factorize(base + 40, np.full_like(base, 23), ids)
    for key in ("cold", "native", "event_only", "within_only", "total_kl"):
        assert np.allclose(f[key], shifted[key])
    assert np.allclose(f["total_kl"], 0)


def test_query_permutation_and_gauge():
    rng = np.random.default_rng(4)
    b = rng.normal(size=(6, 12, 9))
    s, r = M.shared_specific(b)
    perm = np.array([3, 1, 5, 0, 4, 2])
    other, deviations = M.shared_specific(b[perm] + np.arange(6)[:, None, None])
    assert np.allclose(s, other)
    assert np.allclose(r[perm], deviations)
    assert np.allclose(r.mean(0), 0)


def test_finite_components_reconstruct():
    a, b, interaction = M.crossed_components(1, 2, -2, 4)
    assert a + b == 3
    assert interaction == 5


@pytest.mark.parametrize(
    "groups", [np.array([0, 0, 2, 2, 2]), np.zeros(4), np.full(5, np.nan)]
)
def test_invalid_groups(groups):
    base, _ = sample()
    with pytest.raises(ValueError):
        M.factorize(base, base, groups)

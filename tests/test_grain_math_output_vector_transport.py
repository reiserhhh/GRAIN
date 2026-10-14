"""Finite source, gauge and noncommutation witnesses for HM23."""

import numpy as np
import pytest

from grain_research.construction import math_output_vector_transport as M


def test_explicit_feature_primal_dual_and_contraction():
    """A shared kernel does not couple output tasks; linear contraction commutes."""
    rng = np.random.default_rng(2301)
    x, c, d = rng.normal(size=(3, 7, 2))
    feature = np.column_stack((x, np.einsum("ij,ik->ijk", x, c).reshape(7, -1)))
    kernel, w = feature @ feature.T, np.arange(1, 8) / 28
    unit = float(w @ np.diag(kernel))
    f = M.fit_multi(kernel, d, w, unit)
    p = M.apply_multi(kernel, f["weighted_dual"], unit)
    primal = np.linalg.solve(
        feature.T @ (w[:, None] * feature) + unit * np.eye(6),
        feature.T @ (w[:, None] * d),
    )
    np.testing.assert_allclose(p, feature @ primal, atol=1e-14)
    covector = np.array([0.3, -0.2])
    scalar = M.fit_multi(kernel, (d @ covector)[:, None], w, unit)
    np.testing.assert_allclose(
        p @ covector,
        M.apply_multi(kernel, scalar["weighted_dual"], unit)[:, 0],
        atol=1e-14,
    )


def test_reference_nonlinearity_does_not_commute():
    """Scalar fit and complete-vector fit differ for varying actual references."""
    reference = np.array([[2.0, -2.0], [-2.0, 2.0]])
    delta = np.array([[1.0, -1.0], [1.0, -1.0]])
    alphabet, anchor = np.array([0.0, 1.0]), np.zeros(2)
    y, _ = M.readout(reference, delta, alphabet, anchor)
    w, k = np.array([0.5, 0.5]), np.ones((2, 2))
    vector = M.fit_multi(k, delta, w, 1.0)
    scalar = M.fit_multi(k, y[:, None], w, 1.0)
    pv = M.apply_multi(k, vector["weighted_dual"], 1.0)
    py, _ = M.readout(reference, pv, alphabet, anchor)
    ps = M.apply_multi(k, scalar["weighted_dual"], 1.0)[:, 0]
    assert np.max(abs(py - ps)) > 0.003


def test_gauge_alphabet_noop_and_accounts():
    """Coordinate shifts and joint candidate reordering leave the observation intact."""
    rng = np.random.default_rng(2302)
    r, t = rng.normal(size=(2, 5, 4))
    d = M.centered_displacement(r, t)
    np.testing.assert_allclose(d.sum(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(M.centered_displacement(r + 3, t - 7), d, atol=2e-15)
    a, bar = np.arange(4) / 3, np.array([0.1, -0.2, 0.3, -0.2])
    y, parts = M.readout(r, d, a, bar)
    order = [3, 1, 0, 2]
    reordered, _ = M.readout(r[:, order], d[:, order], a[order], bar[order])
    np.testing.assert_allclose(y, reordered, atol=2e-16)
    np.testing.assert_allclose(parts.sum(axis=1), y, atol=1e-16)
    yy, pp = M.readout(r, np.zeros_like(d), a, bar)
    assert not np.any(yy) and not np.any(pp)
    f = M.fit_multi(np.eye(5), d, np.ones(5) / 5, 1.0)
    assert not np.any(M.apply_multi(np.zeros((1, 5)), f["weighted_dual"], 1.0))


def test_source_layout_rejects_held_duplicates_and_missing_queries():
    """The target matrix cannot silently change query source support or role."""
    rows = [
        {"role": "source", "state_index": s, "query": q}
        for s in (2, 4)
        for q in ("a", "b")
    ]
    r = np.zeros((4, 3))
    t = np.arange(12).reshape(4, 3)
    d = M.source_targets(rows, r, t, ["b", "a"], [4, 2])
    np.testing.assert_allclose(d, np.array([[-1, 0, 1, -1, 0, 1]] * 2))
    with pytest.raises(ValueError, match="held"):
        M.source_targets(
            [{**rows[0], "role": "application"}, *rows[1:]], r, t, ["a", "b"], [2, 4]
        )
    with pytest.raises(ValueError, match="support"):
        M.source_targets(rows[:-1], r[:-1], t[:-1], ["a", "b"], [2, 4])
    with pytest.raises(ValueError, match="support"):
        M.source_targets([rows[0], rows[0], *rows[2:]], r, t, ["a", "b"], [2, 4])


def test_remainder_covariance_and_missing_constant_correlation():
    """Nonzero remainders can cancel; constant predictions have no Pearson value."""
    p, q = np.array([-1.0, 0, 1]), np.array([1.0, 0, -1])
    w = np.ones(3) / 3
    v = M.remainder_covariance(np.zeros(3), np.zeros(3), p, q, w)
    assert v["native_covariance"] == 0 and v["PP"] != 0
    assert v["recombination_error"] < 1e-15
    assert M.scalar_metrics(p, np.zeros(3), w)["pearson"] is None


@pytest.mark.parametrize(
    "kernel,target,weights,unit",
    [
        (np.eye(2), np.zeros((2, 3)), np.array([0.5, -0.5]), 1),
        (np.eye(2), np.full((2, 3), np.nan), np.ones(2) / 2, 1),
        (np.array([[1, 2], [0, 1]]), np.zeros((2, 3)), np.ones(2) / 2, 1),
        (np.eye(2), np.zeros((2, 3)), np.ones(2) / 2, 0),
        (-np.eye(2), np.zeros((2, 3)), np.ones(2) / 2, 1),
    ],
)
def test_invalid_source_system(kernel, target, weights, unit):
    """Malformed or indefinite source systems fail before model creation."""
    with pytest.raises(ValueError):
        M.fit_multi(kernel, target, weights, unit)

"""Balanced exchange separates observation functions, not psychological factors."""

import numpy as np
import pytest

from grain_research.construction.math_endpoint_exchange_orbit import (
    balanced_cross,
    exchange,
    inverse_inventory,
    lifted_metric,
    orbit_grams,
    project_actions,
)
from grain_research.construction.math_midpoint_polarization import explicit_observations


def test_involution_and_fixed_midpoint():
    rng = np.random.default_rng(20)
    x, c = rng.normal(size=(9, 4)), rng.normal(size=(9, 4))
    xr, cr = exchange(x, c, 0.7)
    xx, cc = exchange(xr, cr, 0.7)
    np.testing.assert_allclose(xx, x)
    np.testing.assert_allclose(cc, c, atol=1e-15)
    np.testing.assert_allclose(cr + 0.35 * xr, c + 0.35 * x)


def test_explicit_dyad_orbits_and_span_identity():
    rng = np.random.default_rng(32)
    x, c = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    u, v = explicit_observations(x, c, 0.8)
    xr, cr = exchange(x, c, 0.8)
    ur, vr = explicit_observations(xr, cr, 0.8)
    np.testing.assert_allclose(ur, -u - 2 * v)
    np.testing.assert_allclose(vr, v)
    b = orbit_grams(x, c, 0.8)
    orbit, stacked = np.vstack((u, ur)), np.vstack((u, v))
    np.testing.assert_allclose(b["orbit_U_U_reverse"], orbit @ orbit.T, atol=1e-13)
    assert np.linalg.matrix_rank(orbit) == np.linalg.matrix_rank(stacked)
    e, o = project_actions(u, ur)
    np.testing.assert_allclose(e, -v)
    np.testing.assert_allclose(o, u + v)


def test_projectors_are_disjoint_idempotents():
    f, r = np.arange(4.0), np.array([4.0, 6.0, -2.0, 1.0])
    e, o = project_actions(f, r)
    np.testing.assert_array_equal(project_actions(e, e)[0], e)
    np.testing.assert_array_equal(project_actions(e, e)[1], np.zeros(4))
    np.testing.assert_array_equal(project_actions(o, -o)[0], np.zeros(4))
    np.testing.assert_array_equal(project_actions(o, -o)[1], o)
    np.testing.assert_array_equal(e + o, f)
    np.testing.assert_array_equal(e - o, r)


def test_balanced_not_unbalanced_orthogonality():
    original, balanced = balanced_cross([2.0, 3.0], [4.0, 5.0], [0.3, 0.7])
    assert original == pytest.approx(12.9)
    assert abs(balanced) < 1e-14
    # Orthogonality in orbit L2 does not make these ambient dyads orthogonal.
    u, v = explicit_observations([[1.0, 0.0]], [[2.0, 0.0]], 1.0)
    assert (v @ (u + v).T).item() != 0


def test_receiver_defect_and_endpoint_potential():
    rng = np.random.default_rng(91)
    x, c = rng.normal(size=(7, 4)), rng.normal(size=(7, 4))
    a, b, k = rng.normal(size=(4, 4)), rng.normal(size=4), 0.6
    s = (a + a.T) / 2
    law = lambda xx, cc: xx @ b + np.einsum("ij,jk,ik->i", xx, a, cc)
    f = law(x, c)
    e, o = project_actions(f, law(*exchange(x, c, k)))
    v = k / 2 * np.einsum("ij,jk,ik->i", x, s, x)
    np.testing.assert_allclose(e, -v, atol=1e-14)
    np.testing.assert_allclose(o, law(x, c + k * x / 2), atol=1e-14)
    potential = lambda cc: cc @ b / k + np.einsum("ij,jk,ik->i", cc, s, cc) / (2 * k)
    action = potential(c + k * x) - potential(c)
    endpoint = x @ b + np.einsum("ij,jk,ik->i", x, s, c + k * x / 2)
    np.testing.assert_allclose(action, endpoint, atol=1e-14)


def test_skew_is_odd_but_not_conservative():
    a = np.array([[0.0, 1.0], [-1.0, 0.0]])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = np.roll(nodes, -1, axis=0) - nodes
    f = np.einsum("ij,jk,ik->i", x, a, nodes)
    e, _ = project_actions(f, np.einsum("ij,jk,ik->i", -x, a, nodes + x))
    np.testing.assert_allclose(e, 0)
    assert abs(f.sum()) == 1


def test_lifted_metric_and_gauge():
    rho, metric = lifted_metric()
    np.testing.assert_array_equal(rho @ rho, np.eye(2))
    assert not np.allclose(rho.T @ rho, np.eye(2))
    np.testing.assert_array_equal(rho.T @ metric @ rho, metric)
    assert np.linalg.eigvalsh(metric).min() > 0
    rng = np.random.default_rng(16)
    x, c = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    for name, g in orbit_grams(x, c, 0.8).items():
        np.testing.assert_allclose(g, orbit_grams(x @ q, c @ q, 0.8)[name], atol=1e-13)


def test_no_new_orbit_directions_and_zero():
    x = np.eye(3)
    for c, k in [(x, 2.0), (x, 0.0), (np.zeros_like(x), 0.0)]:
        u, v = explicit_observations(x, c, k)
        assert np.linalg.matrix_rank(
            np.vstack((u, -u - 2 * v))
        ) == np.linalg.matrix_rank(u)
    b = orbit_grams(np.zeros((3, 2)), np.ones((3, 2)), 0.7)
    assert all(np.count_nonzero(a) == 0 for a in b.values())


def test_context_inventory_does_not_confuse_swapped_documents():
    desc = [
        {
            "row": 0,
            "context_sha256": "a",
            "reference_sha256": "r",
            "target_sha256": "t",
        },
        {
            "row": 1,
            "context_sha256": "b",
            "reference_sha256": "t",
            "target_sha256": "r",
        },
        {
            "row": 2,
            "context_sha256": "a",
            "reference_sha256": "t",
            "target_sha256": "r",
        },
        {
            "row": 3,
            "context_sha256": "a",
            "reference_sha256": "r",
            "target_sha256": "r",
        },
    ]
    inv = inverse_inventory(desc)
    assert inv[0]["inverse_rows"] == [2]
    assert inv[1]["inverse_rows"] == []
    assert not inv[3]["nontrivial_observed_inverse"]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        exchange([[1.0]], [[np.nan]], 1.0)
    with pytest.raises(ValueError):
        balanced_cross([1.0], [1.0], [0.0])
    with pytest.raises(ValueError):
        project_actions([1.0], [np.nan])

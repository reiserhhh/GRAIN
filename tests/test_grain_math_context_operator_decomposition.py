"""Synthetic exact-dyad, finite-curvature and common-support HM17 tests."""

import numpy as np
import pytest

from grain_research.construction.math_context_operator_decomposition import (
    action_parts,
    compact_operator,
    gram_products,
    potential_value,
    source_basis,
    spectra,
)


def example():
    """Small full-rank dyads independent of real packet endpoints."""
    rng = np.random.default_rng(71)
    return rng.normal(size=(7, 4)), rng.normal(size=(7, 4)), rng.normal(size=(3, 7))


def test_parts_match_explicit_operators():
    x, c, weights = example()
    alpha = weights[0]
    a, b = x.T @ (alpha[:, None] * c), alpha @ x
    s, h = (a + a.T) / 2, (a - a.T) / 2
    p = action_parts(x, c, x, c, alpha, 0.3)
    np.testing.assert_allclose(p["linear"], x @ b, atol=1e-12)
    np.testing.assert_allclose(p["symmetric"], np.sum((x @ s) * c, axis=1), atol=1e-12)
    np.testing.assert_allclose(p["skew"], np.sum((x @ h) * c, axis=1), atol=1e-12)
    np.testing.assert_allclose(
        p["finite_step"], 0.15 * np.sum((x @ s) * x, axis=1), atol=1e-12
    )
    np.testing.assert_allclose(
        p["receiver"] - p["potential"], p["receiver_minus_potential"], atol=1e-12
    )


@pytest.mark.parametrize("part", ["b", "A", "S", "H"])
def test_gram_products_match_explicit_all_query_inner_products(part):
    x, c, weights = example()
    matrices = [x.T @ (alpha[:, None] * c) for alpha in weights]
    if part == "b":
        matrices = [alpha @ x for alpha in weights]
    elif part == "S":
        matrices = [(a + a.T) / 2 for a in matrices]
    elif part == "H":
        matrices = [(a - a.T) / 2 for a in matrices]
    reference = [[np.sum(a * b) for b in matrices] for a in matrices]
    np.testing.assert_allclose(
        gram_products(x, c, weights)[part], reference, atol=1e-11
    )


def test_potential_finite_step_and_telescoping():
    x, c, weights = example()
    alpha, kappa = weights[0], 0.4
    start, end = c, c + kappa * x
    middle = (start + end) / 2
    p = action_parts(x, start, x, c, alpha, kappa)
    phi = [potential_value(z, x, c, alpha, kappa) for z in (start, middle, end)]
    np.testing.assert_allclose(phi[2] - phi[0], p["potential"], atol=1e-12)
    np.testing.assert_allclose(
        (phi[1] - phi[0]) + (phi[2] - phi[1]), p["potential"], atol=1e-12
    )
    reverse = action_parts(-x, end, x, c, alpha, kappa)
    np.testing.assert_allclose(reverse["potential"], -p["potential"], atol=1e-12)


def test_symmetric_operator_still_has_left_endpoint_discretization_error():
    x = np.eye(3)
    c, alpha = x.copy(), np.array([1.0, 2.0, 3.0])
    p = action_parts(x, c, x, c, alpha, 1.0)
    assert np.max(abs(p["skew"])) == 0
    assert np.min(p["finite_step"]) > 0
    np.testing.assert_allclose(p["receiver_minus_potential"], -p["finite_step"])


def test_support_and_operator_spectra_preserve_geometry():
    x, c, weights = example()
    support = source_basis(x, c)
    assert support["rank"] == 4
    assert support["source_reconstruction_max"] < 1e-12
    compact, b = compact_operator(x, c, weights[0], support["basis"])
    full = x.T @ (weights[0, :, None] * c)
    np.testing.assert_allclose(
        support["basis"] @ compact @ support["basis"].T, full, atol=1e-11
    )
    np.testing.assert_allclose(support["basis"] @ b, weights[0] @ x, atol=1e-12)
    s = spectra(compact)
    np.testing.assert_allclose(s["frobenius_squared"]["A"], np.sum(full**2), atol=1e-11)
    assert abs(s["s_h_inner_product"]) < 1e-12
    np.testing.assert_allclose(
        s["frobenius_squared"]["A"],
        s["frobenius_squared"]["S"] + s["frobenius_squared"]["H"],
    )


def test_zero_support_and_zero_displacement():
    x, c, weights = example()
    support = source_basis(np.zeros_like(x), np.zeros_like(c))
    assert support["rank"] == 0 and support["basis"].shape == (4, 0)
    assert spectra(np.zeros((0, 0)))["ranks"] == {"A": 0, "S": 0, "H": 0}
    p = action_parts(np.zeros_like(x), c, x, c, weights[0], 0.4)
    assert all(np.array_equal(v, np.zeros(len(x))) for v in p.values())


def test_isometry_and_context_origin_transport():
    x, c, weights = example()
    alpha, rng = weights[0], np.random.default_rng(72)
    q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    p = action_parts(x, c, x, c, alpha, 0.3)
    rotated = action_parts(x @ q, c @ q, x @ q, c @ q, alpha, 0.3)
    for key in p:
        np.testing.assert_allclose(p[key], rotated[key], atol=1e-11)
    shift = rng.normal(size=4)
    a, b = x.T @ (alpha[:, None] * c), alpha @ x
    moved = x @ (b + a @ shift) + np.sum((x @ a) * (c - shift), axis=1)
    np.testing.assert_allclose(moved, p["receiver"], atol=1e-12)


def test_invalid_dimensions_and_kappa_fail_closed():
    x, c, weights = example()
    with pytest.raises(ValueError):
        action_parts(x, c[:, :2], x, c, weights[0], 0.4)
    with pytest.raises(ValueError):
        action_parts(x, c, x, c, weights[0], 0)


def test_existing_packet_is_not_replaced(tmp_path):
    from scripts.run_grain_math_context_operator_decomposition import prepare

    with pytest.raises(FileExistsError):
        prepare(tmp_path)

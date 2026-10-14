"""Finite Gram/projection identities include designs with and without novelty."""

import numpy as np
import pytest

from grain_research.construction.math_midpoint_polarization import (
    explicit_observations,
    gram_blocks,
    projected_action,
    source_projection,
    spectrum,
)


def test_blocks_and_midpoint_skew():
    rng = np.random.default_rng(72)
    x, c = rng.normal(size=(12, 5)), rng.normal(size=(12, 5))
    kappa = 0.7
    u, v = explicit_observations(x, c, kappa)
    blocks = gram_blocks(x, c, kappa)
    for name, expected in {
        "U": u @ u.T,
        "V": v @ v.T,
        "UV": u @ v.T,
        "W": (u + v) @ (u + v).T,
    }.items():
        np.testing.assert_allclose(blocks[name], expected, atol=1e-13)
    np.testing.assert_allclose(gram_blocks(x, c + kappa * x / 2, 0)["U"], blocks["W"])
    skew = lambda a: np.einsum("ij,ik->ijk", x, a) - np.einsum("ij,ik->ijk", a, x)
    np.testing.assert_allclose(skew(c), skew(c + kappa * x / 2), atol=1e-14)


def test_new_quadratic_direction_and_fixed_action():
    x, c = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    blocks = gram_blocks(x, c, 2.0)
    projection = source_projection(blocks["U"], blocks["UV"], blocks["V"], [1.0])
    assert spectrum(projection["residual"])["rank"] == 1
    assert (
        spectrum(
            np.block([[blocks["U"], blocks["UV"]], [blocks["UV"].T, blocks["V"]]])
        )["rank"]
        == 2
    )
    u, v = explicit_observations(x, c, 2.0)
    operator = np.diag([1.0, 0.0]).reshape(-1)
    predicted = projected_action(x, x, c, 2.0, projection["coefficient"], u @ operator)
    assert predicted[0] == 0
    assert (v @ operator)[0] == 1


def test_no_new_direction_counterexample():
    x = np.eye(3)
    b = gram_blocks(x, x, 2.0)
    p = source_projection(b["U"], b["UV"], b["V"], np.full(3, 1 / 3))
    np.testing.assert_allclose(p["residual"], 0, atol=1e-14)
    assert spectrum(p["residual"])["rank"] == 0


def test_projection_replays_explicit_row_span():
    rng = np.random.default_rng(91)
    x, c = rng.normal(size=(7, 4)), rng.normal(size=(7, 4))
    u, v = explicit_observations(x, c, 0.6)
    b = gram_blocks(x, c, 0.6)
    w = np.arange(1.0, 8.0) / 28
    p = source_projection(b["U"], b["UV"], b["V"], w)
    orthogonal, _ = np.linalg.qr(u.T, mode="reduced")
    remainder = v - (v @ orthogonal) @ orthogonal.T
    np.testing.assert_allclose(p["residual"], remainder @ remainder.T, atol=1e-12)
    operator = rng.normal(size=16)
    expected = (v @ orthogonal) @ (orthogonal.T @ operator)
    np.testing.assert_allclose(
        projected_action(x, x, c, 0.6, p["coefficient"], u @ operator),
        expected,
        atol=1e-12,
    )


def test_orthogonal_gauge():
    rng = np.random.default_rng(15)
    x, c = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    for name, value in gram_blocks(x, c, 0.4).items():
        np.testing.assert_allclose(
            value, gram_blocks(x @ q, c @ q, 0.4)[name], atol=1e-13
        )


def test_zero_and_rank_deficient():
    for x, c in [
        (np.zeros((4, 2)), np.ones((4, 2))),
        (np.ones((4, 2)), np.ones((4, 2))),
    ]:
        b = gram_blocks(x, c, 0.0)
        p = source_projection(b["U"], b["UV"], b["V"], np.full(4, 0.25))
        np.testing.assert_allclose(p["residual"], 0)
        assert np.isfinite(p["coefficient"]).all()
    with pytest.raises(ValueError):
        source_projection(np.eye(2), np.eye(2), np.eye(2), [1.0, 0.0])


def test_signed_eigenvalues_not_silently_clipped():
    value = spectrum(np.diag([1.0, -1e-5]))
    assert value["negative_beyond_tolerance"] == 1
    assert value["eigen_min"] == -1e-5

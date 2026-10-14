#!/usr/bin/env python3
"""Mathematical tests for new moments, integrability and frozen application."""

import numpy as np
import pytest

from grain_research.query_moment_operator import (
    apply_operator,
    cell_residual,
    fit_operator,
    observation_moments,
    symmetric_basis,
    within_direction,
)


def fixture():
    """Repeated exact GRAIN cells with a known symmetric scalar potential."""
    rng = np.random.default_rng(13)
    g = np.repeat(rng.normal(size=(8, 4)), 5, axis=0)
    h = rng.normal(size=(len(g), 7))
    v = np.linalg.qr(rng.normal(size=(7, 3)))[0]
    a = np.diag([0.4, -0.2, 0.1])
    model = {
        "g_mean": g.mean(0),
        "g_scale": g.std(0),
        "query_basis": v,
        "within_matrix": a,
        "beta1": rng.normal(size=(5, 7)),
        "beta2": rng.normal(size=(15, 7)),
        "offsets": np.array([0.1, -0.2, 0.3]),
    }
    return rng, g, h, model


def test_finite_difference_and_scalar_gradient_agree():
    rng, g, h, model = fixture()
    direction = rng.normal(size=h.shape)
    _, fields = apply_operator(g, h, model)
    eps = 1e-5
    difference = (
        apply_operator(g, h + eps * direction, model)[0]
        - apply_operator(g, h - eps * direction, model)[0]
    ) / (2 * eps)
    np.testing.assert_allclose(
        difference, np.einsum("ath,th->a", fields, direction), atol=1e-9
    )


def test_new_moment_changes_while_all_g_cell_mean_moments_are_fixed():
    _, g, h, model = fixture()
    d = within_direction(g, h, model["query_basis"])
    first = observation_moments(g, h, model)
    second = observation_moments(g, h + 0.1 * d, model)
    np.testing.assert_allclose(first["linear"], second["linear"], atol=1e-14)
    np.testing.assert_allclose(first["quadratic"], second["quadratic"], atol=1e-14)
    assert (
        np.linalg.norm(first["within_covariance"] - second["within_covariance"]) > 0.01
    )


def test_cell_residual_orthogonality_and_reordering():
    rng, g, h, _ = fixture()
    r = cell_residual(g, h)
    np.testing.assert_allclose((h - r).T @ r, 0, atol=1e-13)
    order = rng.permutation(len(h))
    np.testing.assert_allclose(r[order], cell_residual(g[order], h[order]))


def test_symmetric_basis_is_orthonormal():
    b = symmetric_basis(4)
    assert b.shape == (10, 4, 4)
    np.testing.assert_allclose(b, b.transpose(0, 2, 1))
    np.testing.assert_allclose(np.einsum("kij,lij->kl", b, b), np.eye(10), atol=1e-15)


def test_source_fit_and_new_application_have_no_teacher_input():
    rng, g, h, model = fixture()
    panels = []
    for _ in range(6):
        h = rng.normal(size=h.shape)
        values, fields = apply_operator(g, h, model)
        panels.append({"g": g, "h": h, "j": fields[2], "value": values[2]})
    fitted = fit_operator(panels, rank=3, alpha=1e-8)
    cold_h = rng.normal(size=h.shape)
    value, field = apply_operator(g, cold_h, fitted)
    assert np.isfinite(value).all() and np.isfinite(field).all()
    truth = apply_operator(g, cold_h, model)[0][2]
    assert abs(value[2] - truth) < 1e-5


def test_distinct_g_singletons_have_no_within_statistic():
    rng, _, h, model = fixture()
    unique_g = rng.normal(size=(len(h), 4))
    m = observation_moments(unique_g, h, model)
    np.testing.assert_array_equal(m["within_covariance"], 0)
    with pytest.raises(ValueError, match="No within"):
        within_direction(unique_g, h, model["query_basis"])


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        cell_residual(np.ones((4, 4)), np.ones((3, 5)))


def test_source_subspace_coordinates_do_not_define_the_score():
    """Orthogonal basis changes leave the physical potential and field unchanged."""
    rng, g, h, model = fixture()
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotated = {
        **model,
        "query_basis": model["query_basis"] @ q,
        "within_matrix": q.T @ model["within_matrix"] @ q,
    }
    for a, b in zip(apply_operator(g, h, model), apply_operator(g, h, rotated)):
        np.testing.assert_allclose(a, b, atol=1e-14)


def test_within_cell_sign_flip_is_not_identified_by_moments():
    """A covariance potential is not a universal, injective text-state reader."""
    _, g, h, model = fixture()
    flipped = h - 2 * cell_residual(g, h)
    assert np.linalg.norm(h - flipped) > 1
    original, other = (
        observation_moments(g, h, model),
        observation_moments(g, flipped, model),
    )
    for key in original:
        np.testing.assert_allclose(original[key], other[key], atol=1e-14)
    np.testing.assert_allclose(
        apply_operator(g, h, model)[0], apply_operator(g, flipped, model)[0], atol=1e-14
    )

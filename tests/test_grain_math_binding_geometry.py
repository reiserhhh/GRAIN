#!/usr/bin/env python3
"""Minimal mathematical transport checks, with a world that can fail."""

import numpy as np

from grain_research.construction import math_binding as B
from grain_research.construction import math_binding_geometry as G


def world():
    """Return nondegenerate role vectors and a fixed source-only program."""
    rng = np.random.default_rng(8)
    a, b = rng.normal(size=(16, 7)), rng.normal(size=(16, 7))
    y = rng.normal(size=10)
    return a, b, B.fit_kernel(a[:10], b[:10], y, alpha=0.1)


def test_independent_role_rotations_preserve_score():
    a, b, model = world()
    qa, qb = G.role_basis(7, 1), G.role_basis(7, 2)
    k = G.metric_kernel(
        G.transform_role(a, qa),
        G.transform_role(b, qb),
        G.transform_role(model["a"], qa),
        G.transform_role(model["b"], qb),
        np.ones(7),
        np.ones(7),
    )
    np.testing.assert_allclose(
        G.frozen_kernel_score(model, k), B.score_kernel(model, a, b)
    )


def test_scaled_basis_requires_metric_transport():
    a, b, model = world()
    qa, qb = G.role_basis(7, 1, True), G.role_basis(7, 2, True)
    roles = (
        G.transform_role(a, qa),
        G.transform_role(b, qb),
        G.transform_role(model["a"], qa),
        G.transform_role(model["b"], qb),
    )
    correct = G.metric_kernel(*roles, qa["scale"] ** -2, qb["scale"] ** -2)
    wrong = G.metric_kernel(*roles, np.ones(7), np.ones(7))
    np.testing.assert_allclose(
        G.frozen_kernel_score(model, correct), B.score_kernel(model, a, b), atol=1e-12
    )
    assert (
        np.max(abs(G.frozen_kernel_score(model, wrong) - B.score_kernel(model, a, b)))
        > 0.01
    )


def test_query_only_basis_change_is_not_transport():
    a, b, model = world()
    k = G.metric_kernel(
        G.transform_role(a, G.role_basis(7, 1)),
        G.transform_role(b, G.role_basis(7, 2)),
        model["a"],
        model["b"],
        np.ones(7),
        np.ones(7),
    )
    assert (
        np.max(abs(G.frozen_kernel_score(model, k) - B.score_kernel(model, a, b)))
        > 0.01
    )


def test_hadamard_requires_a_cross_role_coordinate_identification():
    a, b, _ = world()
    q = G.role_basis(7, 1)
    same = G.hadamard_interaction_gram(G.transform_role(a, q), G.transform_role(b, q))
    original = G.hadamard_interaction_gram(a, b)
    independent = G.hadamard_interaction_gram(
        G.transform_role(a, q), G.transform_role(b, G.role_basis(7, 2))
    )
    np.testing.assert_allclose(same, original)
    assert np.max(abs(independent - original)) > 0.01

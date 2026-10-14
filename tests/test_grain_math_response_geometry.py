#!/usr/bin/env python3
"""Exact response geometry tests and nulls that can falsify blind sharing."""

import numpy as np
import pytest

from grain_research.construction import math_context_operator as C
from grain_research.construction import math_response_geometry as G


def operator(seed=9):
    """Construct a finite operator with no learned psychological labels."""
    rng = np.random.default_rng(seed)
    return {
        "constant": 0.2,
        "source_action": rng.normal(size=(7, 5)),
        "source_object": rng.normal(size=(7, 6)),
        "effective_dual": rng.normal(size=7),
        "action_linear": rng.normal(size=5),
        "object_linear": rng.normal(size=6),
    }


def test_gradient_exact_finite_response():
    rng, op = np.random.default_rng(4), operator()
    a, o, delta = (
        rng.normal(size=(3, 5)),
        rng.normal(size=(3, 6)),
        rng.normal(size=(3, 6)),
    )
    np.testing.assert_allclose(
        np.sum(G.gradient(op, a) * delta, axis=1),
        C.score_compiled(op, a, o + delta) - C.score_compiled(op, a, o),
        atol=1e-12,
    )


def test_opposite_outputs_have_identical_subspaces():
    x = np.array([[1, 0, 0], [0, 2, 0]], float)
    np.testing.assert_allclose(G.principal_cosines(x, -x), [1, 1])
    assert G.overlap(x, -x)["cosine"] == -1
    assert G.transport_metrics(x, -x, 1)["mse_over_zero"] == 4


def test_geometry_and_action_are_orthogonal_gauge_invariant():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(4, 6)), rng.normal(size=(3, 6))
    rot = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    np.testing.assert_allclose(
        G.principal_cosines(x, y), G.principal_cosines(x @ rot, y @ rot), atol=1e-12
    )
    delta = rng.normal(size=(4, 6))
    np.testing.assert_allclose(
        np.sum(x * delta, axis=1), np.sum((x @ rot) * (delta @ rot), axis=1), atol=1e-12
    )


def test_source_null_is_retained_not_deleted():
    basis, _, _ = G.row_basis(np.array([[1, 0, 0], [2, 0, 0]], float))
    delta = np.array([[3, 4, 5]], float)
    supported, unseen = G.support_parts(delta, basis)
    np.testing.assert_allclose(supported + unseen, delta)
    np.testing.assert_allclose(supported @ unseen.T, 0)
    assert unseen[0, 1] == 4


def test_identical_source_actions_cannot_identify_unseen_response():
    x, y = np.array([1, 2, 3]), np.array([2, 4, 6])
    frozen = G.source_transport(x, y)
    assert frozen["beta"] == 2
    assert G.transport_metrics([1], [-2], frozen["beta"])["mse_over_zero"] == 4


def test_zero_source_transport_fails_explicitly():
    with pytest.raises(ValueError):
        G.source_transport([0, 0], [1, 1])


def test_shared_contrast_reconstructs_without_information_claim():
    x, y = np.array([1, 3, -2]), np.array([2, -1, 4])
    shared, contrast = (x + y) / 2, (y - x) / 2
    np.testing.assert_allclose(shared - contrast, x)
    np.testing.assert_allclose(shared + contrast, y)


def test_nan_and_alignment_are_rejected():
    with pytest.raises(ValueError):
        G.row_basis([[float("nan")]])
    with pytest.raises(ValueError):
        G.overlap([1, 2], [1])


def test_small_reference_forces_intersection_without_shared_meaning():
    rng = np.random.default_rng(91)
    x, y = rng.normal(size=(8, 15)), rng.normal(size=(8, 15))
    assert G.principal_cosines(x, y)[0] == pytest.approx(1, abs=1e-12)
    bx, _, _ = G.row_basis(x)
    by, _, _ = G.row_basis(y)
    joined, _, _ = G.row_basis(np.vstack((bx, by)))
    assert len(bx) + len(by) - len(joined) == 1


def test_source_equivalent_functions_can_differ_on_new_objects():
    objects = np.array([[1, 0], [2, 0]], float)
    mean = objects.mean(axis=0)
    basis, _, _ = G.row_basis(objects - mean)
    projected, _ = G.support_parts(objects - mean, basis)
    j = np.array([3, 4], float)
    np.testing.assert_allclose(objects @ j, (mean + projected) @ j)
    cold = np.array([[1.5, 2]], float)
    cold_projected, _ = G.support_parts(cold - mean, basis)
    assert ((cold - mean - cold_projected) @ j).item() == 8

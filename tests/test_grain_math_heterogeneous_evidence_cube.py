#!/usr/bin/env python3
"""Finite composition tests, including positive and false-mechanism worlds."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_heterogeneous_evidence_cube as M


def test_complete_cube_reconstruction_and_parseval():
    rng = np.random.default_rng(123)
    values = rng.normal(size=(8, 4, 2))
    coef = M.coefficients(values, n=3)
    np.testing.assert_allclose(M.evaluate(coef, n=3), values, atol=1e-14)
    assert np.sum(coef**2) == pytest.approx(np.sum(values**2) / 8)
    assert sum(M.order_energy(coef, n=3)["energy"]) == pytest.approx(np.sum(coef**2))


def test_first_order_and_edges_have_declared_orientation():
    x = M.grid(2)
    a = np.array([[0.7, 0.2], [-0.1, 0.5]])
    v = x @ a.T + [3, -2]
    coef = M.coefficients(v, n=2)
    np.testing.assert_allclose(M.linear_operator(coef, n=2), 2 * a)
    np.testing.assert_allclose(
        M.edges(v, n=2), np.stack([np.tile(2 * a[:, j], (2, 1)) for j in range(2)])
    )
    assert M.order_energy(coef, n=2)["interaction_fraction"] < 1e-25


def test_additive_logits_can_create_nonadditive_scores():
    x = M.grid(2)
    z = np.full((4, 1, 10), -10.0)
    z[:, :, 0] = 0
    z[:, 0, 9] = 1 + 1.5 * x[:, 0] + 2 * x[:, 1]
    centered = z - z.mean(-1, keepdims=True)
    coef = M.coefficients(centered, n=2)
    assert M.order_energy(coef, n=2)["interaction_fraction"] < 1e-25
    score = M.decode(z)["score"]
    assert (
        M.order_energy(M.coefficients(score, n=2), n=2)["interaction_fraction"] > 0.01
    )
    np.testing.assert_allclose(M.decode(M.evaluate(coef, order=1, n=2))["score"], score)


def test_true_preanswer_interaction_survives_gauge_removal():
    x = M.grid(2)
    z = np.zeros((4, 1, 10))
    z[:, 0, 9] = x[:, 0] * x[:, 1]
    centered = z - z.mean(-1, keepdims=True)
    c = M.coefficients(centered, n=2)
    assert M.order_energy(c, n=2)["interaction_fraction"] == pytest.approx(1)
    assert (
        np.max(
            np.abs(
                M.decode(z)["score"] - M.decode(M.evaluate(c, order=1, n=2))["score"]
            )
        )
        > 0.01
    )
    np.testing.assert_allclose(
        M.decode(z + np.arange(4)[:, None, None] * 17)["score"], M.decode(z)["score"]
    )


def test_jacobian_agrees_with_finite_differences_and_corner_convexity():
    coef = np.random.default_rng(91).normal(size=(4, 2))
    point = np.array([[0.23, -0.61]])
    j = M.jacobian(coef, point, n=2)[0]
    for k in range(2):
        eps = np.zeros((1, 2))
        eps[0, k] = 1e-5
        difference = (
            M.evaluate(coef, point + eps, n=2) - M.evaluate(coef, point - eps, n=2)
        )[0] / 2e-5
        np.testing.assert_allclose(j[:, k], difference, atol=1e-10)
    weights = np.prod((1 + M.grid(2) * point) / 2, axis=1)
    np.testing.assert_allclose(
        j, np.einsum("c,cij->ij", weights, M.jacobian(coef, M.grid(2), n=2))
    )


def test_injectivity_certificate_has_a_positive_world():
    x = M.grid(2)
    a = np.diag([2.0, 3.0])
    c = M.coefficients(x @ a.T, n=2)
    d = M.injectivity_diagnostic(c, a, n=2)
    assert d["strong_monotonicity_certificate"]
    assert d["source_frame_injectivity_certificate"]
    assert d["source_frame_kappa"] == pytest.approx(0)


def test_failed_certificate_does_not_mean_noninjectivity():
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    c = M.coefficients(M.grid(2) @ rotation.T, n=2)
    d = M.injectivity_diagnostic(c, np.eye(2), n=2)
    assert not d["strong_monotonicity_certificate"]
    assert not d["source_frame_injectivity_certificate"]
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert d["failure_is_not_noninjectivity"]


def test_folded_map_and_singular_source_are_retained():
    x = M.grid(2)
    c = M.coefficients(np.column_stack([x[:, 0] * x[:, 1], x[:, 1]]), n=2)
    np.testing.assert_allclose(M.evaluate(c, [[-0.8, 0], [0.8, 0]], n=2), 0)
    d = M.injectivity_diagnostic(c, np.zeros((2, 2)), n=2)
    assert not d["strong_monotonicity_certificate"]
    assert d["source_frame_condition"] is None
    assert d["source_frame_kappa"] is None


def test_full_qualified_design_has_independent_bits_and_same_order():
    root = Path(__file__).resolve().parents[1]
    material = json.loads(
        (root / "configs/grain_math_passive_event_composition_v1.json").read_text()
    )
    docs = M.documents(material)
    assert len(docs) == len({d["body"] for d in docs}) == 128
    for family in (0, 1):
        chosen = [d for d in docs if d["family"] == family]
        assert {tuple(d["x"]) for d in chosen} == set(
            itertools.product((-1, 1), repeat=6)
        )
        for d in chosen:
            assert len(d["records"]) == 6
            assert [r["event_id"] for r in d["records"]] == [
                f"K{i}" for i in range(1, 7)
            ]
            assert all(r["actor"] == "A" for r in d["records"])
            for i, target in enumerate(M.TARGETS):
                action = material["behaviours"][target][family][int(d["x"][i] > 0)]
                assert (
                    d["body"].splitlines()[i]
                    == f"On occasion K{i + 1}, Person A {action}."
                )


@pytest.mark.parametrize(
    "values,n", [(np.zeros((3, 2)), 2), (np.full((4, 2), np.nan), 2)]
)
def test_incomplete_nonfinite_cubes_fail(values, n):
    with pytest.raises(ValueError):
        M.coefficients(values, n=n)


def test_response_cosine_accepts_vectors_and_interaction_tensors():
    assert M.cosine(np.arange(7), np.arange(7)) == pytest.approx(1)
    assert M.cosine(np.ones((15, 6)), -np.ones((15, 6))) == pytest.approx(-1)
    assert M.cosine(np.zeros((6, 6)), np.eye(6)) is None

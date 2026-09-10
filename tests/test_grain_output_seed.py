#!/usr/bin/env python3
"""Softmax chain rule, binary scale, output gauges and underidentification."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.output_seed import (
    binary_gain,
    coefficient_actions,
    projection_fraction,
    row_span,
    score_seed,
    source_gains,
)


def test_seed_matches_finite_derivative_and_shift_invariance():
    rng = np.random.default_rng(380)
    z, v, w = rng.normal(size=10), np.linspace(0, 1, 10), rng.normal(size=(10, 7))
    result = score_seed(z, v, w)
    eps = 1e-5
    observed = np.array(
        [
            (
                score_seed(z + eps * w[:, k], v, w)["score"]
                - score_seed(z - eps * w[:, k], v, w)["score"]
            )
            / (2 * eps)
            for k in range(7)
        ]
    )
    np.testing.assert_allclose(result["hidden_seed"], observed, atol=1e-10)
    assert abs(result["candidate_seed"].sum()) < 1e-15
    shifted = score_seed(z + 100, v, w + rng.normal(size=7))
    np.testing.assert_allclose(
        shifted["hidden_seed"], result["hidden_seed"], atol=1e-14
    )


def test_binary_seed_is_one_fixed_direction_and_gain():
    w = np.array([[1.0, 3, -1], [-2.0, 5, 4]])
    for z in ([-7, 3], [0, 0], [3, -2]):
        result = score_seed(z, [0, 1], w)
        np.testing.assert_allclose(
            result["hidden_seed"],
            binary_gain(result["score"]) * (w[1] - w[0]),
            atol=1e-15,
        )
    with pytest.raises(ValueError):
        binary_gain(1.1)


def test_same_seed_has_many_compatible_transports():
    rng = np.random.default_rng(381)
    seed = rng.normal(size=5)
    j = rng.normal(size=9)
    transport = np.outer(seed, j) / (seed @ seed)
    null = rng.normal(size=(5, 9))
    null -= np.outer(seed, seed @ null) / (seed @ seed)
    np.testing.assert_allclose((transport + null).T @ seed, j, atol=1e-14)
    probe = rng.normal(size=5)
    assert np.linalg.norm(null.T @ probe) > 1


def test_source_gain_excludes_held_score_and_coefficient_contraction():
    scores = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.8}
    folds = {s: [i for i in range(4) if i != s] for s in range(4)}
    before = source_gains(scores, folds)
    after = source_gains(scores | {0: 0.99}, folds)
    assert before[0] == after[0]
    assert before[1] != after[1]
    rng = np.random.default_rng(382)
    a, v, d = (
        rng.normal(size=(20, 3)),
        rng.normal(size=(3, 8)),
        rng.normal(size=(4, 20, 8)),
    )
    np.testing.assert_allclose(
        coefficient_actions(a, v, d),
        np.einsum("kth,th->k", d, a @ v / len(a)),
        atol=1e-14,
    )


def test_source_span_does_not_expand_with_test_vector():
    basis = row_span([[1, 0, 0], [2, 0, 0]])
    assert len(basis) == 1
    assert projection_fraction([0, 1, 0], basis) == 0
    assert projection_fraction([3, 0, 0], basis) == pytest.approx(1)
    assert projection_fraction([0, 0, 0], basis) is None
    with pytest.raises(ValueError):
        score_seed([0, 1], [1], np.ones((2, 3)))


def test_common_anisotropic_transport_can_change_angles_without_query_switch():
    a, b = np.array([1.0, 0.1]), np.array([1.0, -0.1])
    transform = np.diag([1.0, 10.0])
    seed_cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    transported = (transform @ a, transform @ b)
    assert seed_cosine > 0.98
    assert transported[0] @ transported[1] == 0

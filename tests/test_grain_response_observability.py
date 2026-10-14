#!/usr/bin/env python3
"""Synthetic geometry/observation tests that can reject uninformative designs."""

import numpy as np
import pytest

from grain_research.construction import math_response_observability as M


def test_factorial_reconstruction_and_own_qualification():
    rng = np.random.default_rng(127)
    x = rng.normal(size=(6, 2, 2, 2, 2, 7))
    s = M.sectors(x)
    for o, f in np.ndindex(2, 2):
        reconstructed = sum(
            (2 * o - 1 if owner == "odd" else 1)
            * (1 - 2 * f if focus == "odd" else 1)
            * s[f"actor_{owner}_focus_{focus}"]
            for owner in ("even", "odd")
            for focus in ("even", "odd")
        )
        np.testing.assert_allclose(
            reconstructed, (x[:, :, 1, o, f] - x[:, :, 0, o, f]) / 2
        )
    np.testing.assert_allclose(s["own_A"], (x[:, :, 1, 1, 0] - x[:, :, 0, 1, 0]) / 2)


def test_condition_indexing_direct_block_map():
    f = np.arange(24.0).reshape(6, 2, 2)
    theta = np.arange(12.0)
    direct = np.stack([f[:, c].T @ theta[c::2] for c in range(2)])
    np.testing.assert_allclose(M.matrix(f, indexed=True) @ theta, direct.ravel())


def test_source_null_witness_probability_margins_and_real_information():
    field = np.zeros((6, 2, 6))
    for f in range(6):
        field[f, :, f] = 1
    r = M.matrix(field)
    info, theta = M.ambiguous_law(r)
    assert info["null_dimension"] == 6
    np.testing.assert_allclose(r @ theta, 0, atol=1e-15)
    p, m = M.law_probabilities(theta), M.law_probabilities(-theta)
    np.testing.assert_allclose(p.sum(0), m.sum(0), atol=1e-15)
    np.testing.assert_allclose(p.sum(1), m.sum(1), atol=1e-15)
    assert np.linalg.norm(M.matrix(field, indexed=True) @ theta) > 0


def test_author_expectation_is_not_a_fake_zero_score():
    rng = np.random.default_rng(5)
    minus, plus = rng.normal(size=(12, 6)), rng.normal(size=(12, 6))
    r, origin = ((plus - minus) / 2).T, ((plus + minus) / 2).mean(0)
    _, theta = M.ambiguous_law(r)
    readings = np.stack((minus, plus), axis=1)
    p = M.law_probabilities(theta)
    actual = np.einsum("jb,jbq->q", p, readings)
    np.testing.assert_allclose(actual, origin + r @ theta / 12, atol=1e-14)
    assert np.linalg.norm(origin) > 0


def test_function_metric_invariance_under_invertible_coordinates():
    rng = np.random.default_rng(29)
    d, g = rng.normal(size=(8, 12)), rng.normal(size=(6, 8))
    a = rng.normal(size=(8, 8)) + 8 * np.eye(8)
    r, transformed = g @ d, (g @ np.linalg.inv(a)) @ (a @ d)
    np.testing.assert_allclose(r.T @ r, transformed.T @ transformed, atol=1e-12)


def test_null_world_condition_indexing_cannot_add_missing_feature():
    f = np.ones((6, 2, 6))
    theta = np.array([1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]) * 0.4
    np.testing.assert_allclose(M.matrix(f) @ theta, 0, atol=1e-15)
    np.testing.assert_allclose(M.matrix(f, indexed=True) @ theta, 0, atol=1e-15)


def test_zero_and_shape_fail_closed():
    info, theta = M.ambiguous_law(np.eye(6))
    assert info["status"] == "NO_SOURCE_AMBIGUITY"
    assert not theta.any()
    report = M.correspondence(np.zeros((6, 2, 4)), np.zeros((6, 2, 4)))
    assert report["matching_rank"] is None
    with pytest.raises(ValueError):
        M.sectors(np.zeros((6, 2, 2, 2, 7)))

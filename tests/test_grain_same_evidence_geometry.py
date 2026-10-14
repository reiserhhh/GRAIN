#!/usr/bin/env python3
"""Synthetic algebra tests; no source text, model or private records required."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_same_evidence_geometry as M


def test_character_reconstruction_and_linear_compression():
    rng = np.random.default_rng(137)
    x, d = rng.normal(size=(2, 2, 11)), rng.normal(size=(11, 4))
    np.testing.assert_allclose(M.reconstruct(M.characters(x)), x, atol=1e-14)
    np.testing.assert_allclose(M.characters(x @ d), M.characters(x) @ d, atol=1e-14)


def test_cka_rotations_and_uniform_scaling():
    rng = np.random.default_rng(138)
    x, y = rng.normal(size=(6, 8)), rng.normal(size=(6, 5))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    a = M.gram_alignment(M.centered_gram(x), M.centered_gram(y))
    b = M.gram_alignment(M.centered_gram(-3 * x @ q), M.centered_gram(y))
    assert abs(a - b) < 1e-14
    assert abs(M.gram_alignment(M.centered_gram(x), M.centered_gram(x)) - 1) < 1e-14


def test_constant_field_is_not_perfect_alignment():
    x = np.ones((6, 3))
    assert M.gram_alignment(M.centered_gram(x), M.centered_gram(x)) is None
    result, values = M.finite_pairings(x, x)
    assert result["cka"] is None and len(values) == 0


def test_all_permutations_and_direct_formula():
    rng = np.random.default_rng(139)
    x, y = rng.normal(size=(6, 3)), rng.normal(size=(6, 5))
    actual, values = M.finite_pairings(x, y)
    assert len(values) == 720
    direct = []
    xc = x - x.mean(axis=0)
    for p in itertools.permutations(range(6)):
        yc = y[list(p)] - y.mean(axis=0)
        direct.append(
            np.linalg.norm(xc.T @ yc) ** 2
            / (np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc))
        )
    np.testing.assert_allclose(values, direct, atol=1e-14)
    assert abs(actual["cka"] - values[0]) < 1e-14


def test_account_gap_and_compression_counterexample():
    x = np.array([[[b, c, b * c] for c in (-1, 1)] for b in (-1, 1)], float)
    np.testing.assert_allclose(M.account_gap(x), [0, 0, 0.8], atol=1e-14)
    np.testing.assert_allclose(M.account_gap(x), 0.8 * M.characters(x)[3], atol=1e-14)
    # Marginal coordinates may miss the joint action, without the text lacking it.
    np.testing.assert_allclose(M.account_gap(x[..., :2]), 0, atol=1e-14)


@pytest.mark.parametrize("bad", [np.zeros((3, 2, 4)), np.full((2, 2, 1), np.nan)])
def test_incomplete_or_nonfinite_support_rejected(bad):
    with pytest.raises(ValueError):
        M.characters(bad)


def test_invalid_account_law_rejected():
    with pytest.raises(ValueError):
        M.account_gap(np.ones((2, 2, 1)), 1)

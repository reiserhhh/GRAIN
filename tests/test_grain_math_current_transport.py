#!/usr/bin/env python3
"""Numerical mechanism checks with synthetic arrays and explicit null worlds."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_current_transport as M
from grain_research.construction.math_cycle_capacity import cycles


def test_positive_source_interpolation_and_nonlinear_rank():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(M.probability_map(x, x), np.eye(4))
    t = M.probability_map(x, x + 0.2)
    assert t.min() > 0
    np.testing.assert_allclose(t.sum(1), 1)
    assert np.linalg.matrix_rank(M.centered_map_spectrum(t)[0], tol=1e-10) == 3


def test_law_duality_stationarity_and_gradient_null_rectangular():
    rng = np.random.default_rng(981)
    t = rng.uniform(size=(7, 5))
    t /= t.sum(1, keepdims=True)
    d, plus, minus, _ = M.fundamental_currents(7)
    f = rng.normal(size=(5, 5, 3))
    potential = rng.normal(size=(5, 3))
    gradient = potential[:, None] - potential[None, :]
    for current, p, m in zip(d, plus, minus, strict=True):
        dp = M.pushforward(current, t)
        pp, mp = M.pushforward(p, t), M.pushforward(m, t)
        assert min(pp.min(), mp.min()) >= 0
        np.testing.assert_allclose(pp.sum(), 1)
        np.testing.assert_allclose(pp.sum(1), pp.sum(0), atol=1e-13)
        np.testing.assert_allclose(pp.sum(1), mp.sum(1), atol=1e-13)
        np.testing.assert_allclose(pp + pp.T, mp + mp.T, atol=1e-13)
        np.testing.assert_allclose(dp.sum(1), 0, atol=1e-13)
        np.testing.assert_allclose(dp, -dp.T, atol=1e-13)
        np.testing.assert_allclose(
            np.einsum("ij,ijq->q", current, M.pullback(f, t)),
            np.einsum("ij,ijq->q", dp, f),
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.einsum("ij,ijq->q", current, M.pullback(gradient, t)), 0, atol=1e-13
        )


def test_uniform_map_annihilates_current_but_identity_keeps_it():
    d, _, _, _ = M.fundamental_currents(6)
    f = np.moveaxis(d[:3], 0, -1)
    t = np.full((6, 6), 1 / 6)
    np.testing.assert_allclose(cycles(M.pullback(f, t))[2], 0, atol=1e-13)
    np.testing.assert_array_equal(M.pullback(f, np.eye(6)), f)


def test_exterior_singular_products_match_actual_current_map():
    rng = np.random.default_rng(982)
    t = rng.uniform(size=(5, 6))
    t /= t.sum(1, keepdims=True)
    h0, _, _ = M.centered_map_spectrum(t)
    a = np.linalg.eigh(np.eye(5) - 1 / 5)[1][:, 1:]
    b = np.linalg.eigh(np.eye(6) - 1 / 6)[1][:, 1:]
    linear = a.T @ h0 @ b
    s = np.linalg.svd(linear, compute_uv=False)
    pairs_a = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    pairs_b = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    exterior = np.array(
        [
            [
                linear[i, k] * linear[j, l] - linear[i, l] * linear[j, k]
                for k, l in pairs_b
            ]
            for i, j in pairs_a
        ]
    )
    expected = sorted([s[i] * s[j] for i, j in pairs_a], reverse=True)
    np.testing.assert_allclose(np.linalg.svd(exterior, compute_uv=False), expected)


def test_permutation_equivariance_and_observer_change_counterexample():
    rng = np.random.default_rng(983)
    x = rng.normal(size=(6, 3))
    y = rng.normal(size=(7, 3))
    f = rng.normal(size=(6, 6, 2))
    perm = [3, 1, 5, 0, 2, 4]
    t = M.probability_map(x, y)
    ft = M.pullback(f, t)
    np.testing.assert_allclose(
        M.pullback(f[perm][:, perm], M.probability_map(x[perm], y)), ft
    )
    # Even exact semantic addresses cannot transport a changed observer law.
    fc = cycles(f)[2]
    native = -fc
    oracle = M.pullback(fc, np.eye(6))
    assert np.sum(oracle * native) < 0


def test_reject_duplicate_points_or_nonprobability_mapping():
    with pytest.raises(ValueError):
        M.probability_map(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        M.checked_map(np.eye(3) * 2)
    with pytest.raises(ValueError):
        M.checked_map(np.eye(3) - 0.1)


def test_cold_cli_opens_only_declared_data_and_preserves_source(tmp_path):
    rng = np.random.default_rng(984)
    source = {"kernel": rng.normal(size=(2, 18, 18, 18))}
    application = {}
    for context in range(2):
        for bank, dimension in (("factor4", 4), ("svd24", 24)):
            key = f"points_{context}_{bank}"
            source[key] = rng.normal(size=(18, dimension))
            application[key] = source[key].copy()
    bundle, input_file, prediction, audit = [
        tmp_path / name
        for name in ("bundle.npz", "input.npz", "output.npz", "audit.json")
    ]
    np.savez_compressed(bundle, **source)
    np.savez_compressed(input_file, **application)
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/score_grain_current_transport_cold.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            *map(str, (bundle, input_file, prediction, audit)),
        ],
        check=True,
    )
    result = np.load(prediction, allow_pickle=False)
    for context in range(2):
        for bank in ("factor4", "svd24"):
            np.testing.assert_array_equal(result[f"map_{context}_{bank}"], np.eye(18))
            np.testing.assert_array_equal(
                result[f"kernel_{context}_{bank}"], source["kernel"][context]
            )
    record = json.loads(audit.read_text())
    assert set(record["opened_paths"]) == {
        str(p.resolve()) for p in (bundle, input_file, prediction, audit)
    }
    assert not record["application_target_tables_queries_truth_or_text"]

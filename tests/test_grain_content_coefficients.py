#!/usr/bin/env python3
"""Weighted source fitting and target-free local coefficient reconstruction."""

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.vector_mechanisms.content_coefficients import (
    author_weights,
    coefficient_field,
    fit_head,
    fit_preprocessor,
    moments,
    predict_coefficients,
    raw_design,
)


def panel(seed=34):
    """A bilinear coefficient world with independent source contexts."""
    rng = np.random.default_rng(seed)
    h = [rng.normal(size=(n, 7)) for n in (160, 190, 210)]
    g = [rng.normal(size=(len(x), 4)) for x in h]
    p = fit_preprocessor(h, g, rank=6)
    basis = np.eye(7)[:2]
    w = np.zeros((raw_design(h[0], g[0], p, True).shape[1], 2))
    w[-24, 0], w[-23, 1] = 2, -3
    j = [raw_design(x, z, p, True) @ w @ basis / len(x) for x, z in zip(h, g)]
    return h, g, j, p, basis, w


def test_author_mass_and_constant_scaling():
    w = author_weights([10, 100])
    np.testing.assert_allclose([w[:10].sum(), w[10:].sum()], [0.5, 0.5])
    x = np.column_stack([np.r_[np.full(10, 2), np.full(100, 6)], np.ones(110)])
    m, s, c = moments(x, w)
    np.testing.assert_allclose(m, [4, 1])
    np.testing.assert_allclose(s, [2, 1])
    assert c.tolist() == [False, True]


def test_pca_center_and_projected_scales_use_source_mass():
    h, g, _, p, _, _ = panel()
    w = author_weights([len(x) for x in h])
    np.testing.assert_allclose(p["h_center"], w @ np.concatenate(h), atol=1e-13)
    x = np.concatenate([raw_design(a, b, p, False)[:, 3:] for a, b in zip(h, g)])
    np.testing.assert_allclose(w @ x, 0, atol=1e-12)
    np.testing.assert_allclose(w @ x**2, 1, atol=1e-12)


def test_bilinear_source_model_recovers_known_held_coefficients():
    h, g, j, p, v, w = panel()
    m, s = fit_head(h, g, j, p, v, True, alpha=0)
    rng = np.random.default_rng(39)
    x, z = rng.normal(size=(81, 7)), rng.normal(size=(81, 4))
    a = predict_coefficients(x, z, m)
    np.testing.assert_allclose(a, raw_design(x, z, p, True) @ w, atol=1e-11)
    np.testing.assert_allclose(coefficient_field(a, v), a @ v / len(a))
    assert s["source_coefficient_error"] < 1e-20
    simple, _ = fit_head(h, g, j, p, v, False)
    assert np.mean((predict_coefficients(x, z, simple) - a) ** 2) > 0.1


def test_saved_model_prediction_ignores_evaluator_targets():
    h, g, j, _p, v, _ = panel()
    model, _ = fit_head(
        h[:2], g[:2], j[:2], fit_preprocessor(h[:2], g[:2], rank=6), v, True
    )
    b = io.BytesIO()
    np.savez(b, **model)
    b.seek(0)
    with np.load(b, allow_pickle=False) as z:
        saved = dict(z)
    first = predict_coefficients(h[2], g[2], saved)
    j[2][:] = 1e6
    np.testing.assert_array_equal(first, predict_coefficients(h[2], g[2], saved))
    np.testing.assert_array_equal(first, predict_coefficients(h[2], g[2], model))


def test_content_arm_independent_of_grain_and_input_shapes_checked():
    h, g, j, p, v, _ = panel()
    model, _ = fit_head(h, g, j, p, v, False)
    np.testing.assert_array_equal(
        predict_coefficients(h[0], g[0], model),
        predict_coefficients(h[0], 20 * g[0], model),
    )
    with pytest.raises(ValueError):
        predict_coefficients(h[0], g[0][:-1], model)
    with pytest.raises(ValueError):
        fit_head(h, g, [j[0][:-1], *j[1:]], p, v, True)
    with pytest.raises(ValueError):
        fit_preprocessor(h, g, rank=7)


def test_intercept_is_unpenalized():
    h, g, _, p, v, _ = panel()
    j = [np.tile([2.0, -3.0], (len(x), 1)) @ v / len(x) for x in h]
    m, _ = fit_head(h, g, j, p, v, True, alpha=100)
    np.testing.assert_allclose(
        predict_coefficients(h[0], g[0], m),
        np.tile([2.0, -3.0], (len(h[0]), 1)),
        atol=1e-12,
    )


def test_independent_runtime_denies_gradient_packet(tmp_path):
    h, g, j, p, v, _ = panel()
    model, _ = fit_head(h, g, j, p, v, True)
    mp, ip, gp, op = [
        tmp_path / (name + ".npz") for name in ("model", "inputs", "gradient", "output")
    ]
    np.savez(mp, **model)
    np.savez(ip, hidden=h[0], grain=g[0], positions=np.arange(len(h[0])))
    np.savez(gp, gradient=j[0])
    job = tmp_path / "jobs.json"
    job.write_text(json.dumps([{"key": "held", "model": str(mp), "inputs": str(ip)}]))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/replay_grain_content_coefficients.py"),
            "--jobs",
            str(job),
            "--output",
            str(op),
            "--forbid",
            str(gp),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {
        "jobs": 1,
        "denial_probes": 1,
        "extra_denials": 0,
    }
    with np.load(op) as z:
        np.testing.assert_allclose(
            z["held"], predict_coefficients(h[0], g[0], model), atol=1e-12
        )

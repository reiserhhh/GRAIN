#!/usr/bin/env python3
"""Low-variance query directions, frozen base columns and target-free inference."""

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.vector_mechanisms.content_coefficients import (
    author_weights,
    fit_head,
    fit_preprocessor,
    predict_coefficients,
)
from grain_research.vector_mechanisms.position_reader import position_design
from grain_research.vector_mechanisms.query_residual_input import (
    base_design,
    coordinate_support,
    covariance_basis,
    fit_extension,
    predict_extension,
    residual_hidden,
    variance_basis,
)


def world():
    """A signal in a low-variance coordinate, outside one dominant PCA direction."""
    rng = np.random.default_rng(350)
    x = rng.normal(size=(600, 7))
    nuisance = np.tile(position_design(200, 3), (3, 1))
    q = np.linalg.qr(nuisance)[0]
    x -= q @ (q.T @ x)
    x = np.linalg.qr(x)[0] * np.sqrt(len(x))
    h = x * np.array([100, 10, 8, 6, 4, 2, 0.01])
    hs = list(np.split(h, 3))
    gs = [rng.normal(size=(len(a), 4)) for a in hs]
    y = np.column_stack([x[:, -1], 2 * x[:, -1]])
    v = np.eye(7)[:2]
    js = [a @ v / len(a) for a in np.split(y, 3)]
    prep = fit_preprocessor(hs, gs, rank=1)
    base, _ = fit_head(hs, gs, js, prep, v, False)
    return hs, gs, js, base


def test_query_finds_low_variance_direction_missed_by_variance_control():
    hs, gs, js, base = world()
    variance = variance_basis(hs, base, rank=1)
    query, qm = fit_extension(hs, gs, js, base)
    control, vm = fit_extension(hs, gs, js, base, variance)
    assert qm["extra_rank"] == 1
    assert abs(query["extra_basis"][0, -1]) > 0.99
    assert abs(variance["extra_basis"][0, 1]) > 0.99
    assert qm["source_coefficient_error"] < 0.001
    assert vm["source_coefficient_error"] > 0.8
    assert qm["extra_residual_energy_fraction"] < vm["extra_residual_energy_fraction"]
    np.testing.assert_allclose(query["extra_basis"] @ base["h_basis"].T, 0, atol=1e-12)
    rng = np.random.default_rng(359)
    h = rng.normal(size=(170, 7)) * np.array([100, 10, 8, 6, 4, 2, 0.01])
    g = rng.normal(size=(len(h), 4))
    target = h[:, -1, None] / 0.01 * np.array([1.0, 2.0])
    assert (
        np.sum((predict_extension(h, g, query) - target) ** 2) / np.sum(target**2)
        < 0.001
    )
    assert (
        np.sum((predict_extension(h, g, control) - target) ** 2) / np.sum(target**2)
        > 0.8
    )


def test_zero_cross_covariance_adds_no_directions():
    hs, gs, _, base = world()
    r = np.concatenate([residual_hidden(h, base) for h in hs])
    w = author_weights([len(h) for h in hs])
    z = covariance_basis(r, np.zeros((len(r), 2)), w, base["h_basis"])
    assert z["extra_basis"].shape == (0, 7)
    js = [np.zeros((len(h), 7)) for h in hs]
    # With rank0, a nonzero constant target still has a valid finite predictor.
    js = [j + np.array([1.0, 2.0, 0, 0, 0, 0, 0]) / len(j) for j in js]
    m, _ = fit_extension(hs, gs, js, base, z)
    assert coordinate_support(hs[0], m)["held_coordinate_outside_fraction"] == 0
    assert np.isfinite(predict_extension(hs[0], gs[0], m)).all()


def test_base_columns_and_saved_prediction_are_unchanged_by_evaluator_target():
    hs, gs, js, base = world()
    model, _ = fit_extension(hs, gs, js, base)
    np.testing.assert_array_equal(
        base_design(hs[0], gs[0], model), base_design(hs[0], gs[0], base)
    )
    np.testing.assert_allclose(
        base_design(hs[0], gs[0], base) @ base["weights"],
        predict_coefficients(hs[0], gs[0], base),
    )
    buf = io.BytesIO()
    np.savez(buf, **model)
    buf.seek(0)
    with np.load(buf) as z:
        restored = dict(z)
    y = predict_extension(hs[0], gs[0], restored)
    js[0][:] = 999
    np.testing.assert_array_equal(y, predict_extension(hs[0], gs[0], restored))
    np.testing.assert_array_equal(y, predict_extension(hs[0], gs[0], model))
    with pytest.raises(ValueError):
        predict_extension(hs[0], gs[0][:-1], model)


def test_independent_residual_runtime_uses_only_fixed_assets(tmp_path):
    hs, gs, js, base = world()
    model, _ = fit_extension(hs, gs, js, base)
    mp, ip, gp, op = [
        tmp_path / (n + ".npz") for n in ("model", "input", "gradient", "output")
    ]
    np.savez(mp, **model)
    np.savez(ip, hidden=hs[0], grain=gs[0], positions=np.arange(len(hs[0])))
    np.savez(gp, gradient=js[0])
    job = tmp_path / "jobs.json"
    job.write_text(json.dumps([{"key": "held", "model": str(mp), "inputs": str(ip)}]))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/replay_grain_query_residual.py"),
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
            z["held"], predict_extension(hs[0], gs[0], model), atol=1e-12
        )

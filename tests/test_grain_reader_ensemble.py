#!/usr/bin/env python3
"""Frozen ensemble gauge equivalence and numeric-only runtime checks."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.vector_mechanisms.query_residual_input import (
    fit_extension,
    predict_extension,
)
from grain_research.vector_mechanisms.reader_ensemble import (
    ensemble_field,
    pack_models,
    predict_ensemble_coefficients,
    unpack_models,
)
from tests.test_grain_query_residual_input import world


def ensemble_world():
    """Four equivalent outputs in deliberately different coordinate gauges."""
    hs, gs, js, base = world()
    model, _ = fit_extension(hs, gs, js, base)
    models = []
    for angle in (0, 0.6, 2.1, 3.0):
        r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        models.append(
            dict(
                model, basis=r @ model["basis"], aug_weights=model["aug_weights"] @ r.T
            )
        )
    return hs[0], gs[0], models


def test_equal_fields_not_equal_raw_coefficients():
    h, g, models = ensemble_world()
    packed = pack_models(models)
    a = predict_ensemble_coefficients(h, g, packed)
    expected = sum(predict_extension(h, g, m) @ m["basis"] for m in models) / (
        4 * len(h)
    )
    np.testing.assert_allclose(ensemble_field(a, packed), expected, atol=1e-15)
    np.testing.assert_allclose(
        expected,
        predict_extension(h, g, models[0]) @ models[0]["basis"] / len(h),
        atol=1e-15,
    )
    wrong = (
        np.mean(np.stack(np.split(a, 4, axis=1)), axis=0) @ models[0]["basis"] / len(h)
    )
    assert np.linalg.norm(wrong - expected) > 0.01
    with pytest.raises(ValueError):
        pack_models(models[:3])
    with pytest.raises(ValueError):
        ensemble_field(a[:, :-1], packed)
    assert len(unpack_models(packed)) == 4


def test_packed_assets_and_restricted_replay(tmp_path):
    h, g, models = ensemble_world()
    packed = pack_models(models)
    mp, ip, gp, op = [
        tmp_path / (name + ".npz") for name in ("model", "input", "gradient", "output")
    ]
    np.savez(mp, **packed)
    np.savez(ip, hidden=h, grain=g, positions=np.arange(len(h)))
    np.savez(gp, forbidden=np.ones_like(h))
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"key": "held", "model": str(mp), "inputs": str(ip)}]))
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/replay_grain_reader_ensemble.py"
            ),
            "--jobs",
            str(jobs),
            "--output",
            str(op),
            "--forbid",
            str(gp),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {
        "jobs": 1,
        "denial_probes": 1,
        "extra_denials": 0,
    }
    with np.load(op) as z:
        np.testing.assert_allclose(
            z["held"], predict_ensemble_coefficients(h, g, packed), atol=1e-12
        )

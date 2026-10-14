"""Anchor identifiability, population moment recovery and text-only scoring."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f11_anchor import (
    covariance_basis,
    fit_anchor,
    text_only_blup,
)


def test_covariance_rank_gains_one_with_anchor():
    assert np.linalg.matrix_rank(covariance_basis(False)) == 8
    assert np.linalg.matrix_rank(covariance_basis(True)) == 9


def test_exact_population_moments_recover_components():
    coeff = np.array([1.0, 0.3, 1.0, 0.3, 0.3, 1.0, 0.25, 0.25, 0.25])
    basis = covariance_basis(True)
    covariance = (basis @ coeff).reshape(21, 21)
    covariance[18:, 18:] += 0.25 * np.eye(3)
    # Centered +/- Cholesky columns yield this exact unbiased sample covariance.
    root = np.linalg.cholesky(covariance)
    rows = np.vstack([root.T, -root.T]) * np.sqrt(41 / 2)
    fit = fit_anchor(rows[:, :18], rows[:, 18:])
    np.testing.assert_allclose(fit["raw_coefficients"], coeff, atol=1e-12)


def test_application_scoring_has_no_anchor_input():
    model = {
        "sigma": np.eye(3),
        "method": 0.25,
        "error": 0.25,
        "actor": 0.25,
        "mean": np.zeros(18),
    }
    p, v = text_only_blup(model, np.zeros((4, 18)))
    np.testing.assert_array_equal(p, np.zeros((4, 3)))
    assert np.linalg.eigvalsh(v).min() > 0


def test_both_worlds_smoke_complete(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "anchor"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_grain_gptpro_f11_anchor.py"),
            "--smoke",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads((output / "summary.json").read_text())["prediction_rows"] == 216

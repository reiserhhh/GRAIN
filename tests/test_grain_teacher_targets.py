#!/usr/bin/env python3
"""Target information tests independent of private author data."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.target_accounting import (
    native_chain,
    projection_account,
    projection_draws,
)
from scripts.analyze_grain_teacher_targets import join_panel, verify_digest


def orthogonal_columns():
    """Return centered orthogonal finite author coordinates."""
    rng = np.random.default_rng(17)
    x = rng.normal(size=(60, 3))
    q, _ = np.linalg.qr(x - x.mean(0))
    return q.T


def test_affine_shrinkage_is_not_missing_criterion_information():
    x, _, _ = orthogonal_columns()
    account = projection_account(4 * x + 2, x, x)
    assert account["total_error_criterion_contribution"] == pytest.approx(0.75)
    assert account["amplitude_contribution"] == pytest.approx(0.75)
    assert abs(account["non_affine_contribution"]) < 1e-14
    assert account["aligned_contribution"] == pytest.approx(1)
    assert account["projection_identity_error"] < 1e-14


def test_omitted_orthogonal_component_has_criterion_link():
    x, z, _ = orthogonal_columns()
    account = projection_account(x + z, x, z)
    assert account["non_affine_contribution"] == pytest.approx(1 / np.sqrt(2))
    assert abs(account["aligned_contribution"]) < 1e-14


def test_perfect_imitation_does_not_imply_criterion_accuracy():
    x, z, _ = orthogonal_columns()
    account = projection_account(x, x, z)
    assert account["teacher_fidelity_r"] == pytest.approx(1)
    assert abs(account["proxy_r"]) < 1e-14


def test_non_affine_error_does_not_prove_extra_input_information():
    x = np.linspace(-1, 1, 61)
    target = x**2
    account = projection_account(target, x, target)
    assert account["non_affine_contribution"] == pytest.approx(1)
    # A nonlinear function of the same scalar recovers the target exactly.
    np.testing.assert_allclose(x**2, target)


def test_chain_reconstruction_and_signed_cancellation():
    x, z, h = orthogonal_columns()
    account = native_chain(x, 2 * x, x + z, z, h, x, same_observer=3 * x)
    np.testing.assert_allclose(account["values"].sum(1), x, atol=1e-15)
    assert account["contributions"][0] == pytest.approx(-2)
    assert account["closure_error"] < 1e-14
    with pytest.raises(ValueError):
        native_chain(x, z[:-1], x, z, h, x)


def test_vectorized_bootstrap_refits_alignment_each_draw():
    x, z, h = orthogonal_columns()
    t, p, y = x + z, x + 0.3 * h, z + 0.1 * h
    indices = np.random.default_rng(4).integers(0, len(x), (6, len(x)))
    vectorized = projection_draws(t, p, y, indices)
    for i, idx in enumerate(indices):
        row = projection_account(t[idx], p[idx], y[idx])
        for key, values in vectorized.items():
            assert values[i] == pytest.approx(row[key], abs=1e-14)
    with pytest.raises(ValueError):
        projection_draws(t, p, y, indices.astype(float))


def panel():
    """Tiny three-view panel with arbitrary IDs, not real people."""
    fixed = pd.DataFrame(
        {
            "author_id": ["a"],
            "query": ["TF_cont"],
            "score": [0.4],
            "configuration": [0.2],
            "mean": [0.1],
            "final": [0.1],
        }
    )
    criteria = pd.DataFrame(
        {"author_id": ["a"], "query": ["TF_cont"], "y_true": [1], "prediction": [0.5]}
    )
    observer = pd.DataFrame(
        {
            "author_id": ["a"],
            "query": ["TF_cont"],
            "old_teacher": [0.5],
            "new_teacher": [0.45],
        }
    )
    oof = pd.DataFrame(
        [
            {
                "author_id": "a",
                "query": "TF_cont",
                "view": v,
                "arm": "pooled",
                "fold": 0,
                "actual_c": 0.2,
                "actual_r": 0.1,
                "predicted_c": 0.3,
                "predicted_r": 0.15,
            }
            for v in range(3)
        ]
    )
    return oof, fixed, criteria, observer


def test_join_rejects_duplicate_missing_and_cross_fold_rows():
    oof, fixed, criteria, observer = panel()
    assert len(join_panel(oof, fixed, criteria, observer)) == 1
    with pytest.raises(ValueError):
        join_panel(oof.iloc[:2], fixed, criteria, observer)
    with pytest.raises(ValueError):
        join_panel(oof, fixed, pd.concat([criteria, criteria]), observer)
    with pytest.raises(ValueError):
        join_panel(oof, fixed, criteria.iloc[:0], observer)
    oof.loc[0, "fold"] = 1
    with pytest.raises(ValueError):
        join_panel(oof, fixed, criteria, observer)


def test_changed_configuration_or_observer_cannot_be_silently_joined():
    oof, fixed, criteria, observer = panel()
    fixed.loc[0, "configuration"] = 0.3
    with pytest.raises(ValueError):
        join_panel(oof, fixed, criteria, observer)
    fixed.loc[0, "configuration"] = 0.2
    observer.loc[0, "old_teacher"] = 0.4
    with pytest.raises(ValueError):
        join_panel(oof, fixed, criteria, observer)


def test_changed_frozen_bytes_stop_analysis(tmp_path):
    path = tmp_path / "prediction.csv"
    path.write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_digest(path, "wrong-digest")

#!/usr/bin/env python3
"""Synthetic correctness tests for the one-candidate source-span experiment."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.source_span import (
    author_means,
    fit_linear,
    predict_linear,
)
from grain_research.vector_mechanisms.state_geometry import (
    decompose_state_error,
    residual_basis,
)
from grain_research.vector_mechanisms.state_transport import fit_transport, reconstruct


def test_linear_extension_exactly_matches_previous_linear_convention():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(8, 12))
    y = x + rng.normal(size=(8, 12))
    a, b = fit_linear(x, y), fit_transport(x, y)
    for key in ("mean_x", "scale", "train_x", "mean_r", "col1", "grand1", "coef1"):
        np.testing.assert_allclose(a[key], b[key])
    z = rng.normal(size=12)
    np.testing.assert_allclose(predict_linear(a, z), reconstruct(b, z)["transport1"])
    assert "coef2" not in a


def test_reference_aggregation_weights_authors_once_and_pairs_views():
    rows = [
        {
            "role": "reference",
            "author_id": author,
            "view": view,
            "x": [value, value + 1],
            "y": [value + 2, value + 3],
        }
        for author, view, value in (("b", 1, 6), ("a", 0, 0), ("b", 0, 2), ("a", 1, 4))
    ]
    ids, x, y = author_means(rows)
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(x, [[2, 3], [4, 5]])
    np.testing.assert_array_equal(y, [[4, 5], [6, 7]])
    for bad in (
        rows[:-1],
        rows + [rows[0]],
        [{**r, "role": "evaluation"} for r in rows],
    ):
        with pytest.raises(ValueError):
            author_means(bad)


def test_expanding_affine_snapshot_span_cannot_increase_oracle_state_error():
    rng = np.random.default_rng(47)
    x = rng.normal(size=(12, 40))
    residual = rng.normal(size=(12, 5)) @ rng.normal(size=(5, 40))
    residual[6:] += rng.normal(size=(6, 3)) @ rng.normal(size=(3, 40))
    y = x + residual
    old, new = fit_linear(x[:6], y[:6]), fit_linear(x, y)
    z, actual = rng.normal(size=(2, 40))
    energies = []
    for model in (old, new):
        est = predict_linear(model, z)
        _, _, _, stats = decompose_state_error(
            model, residual_basis(model), z, actual, est
        )
        assert stats["orthogonality_accounting_error"] < 1e-9
        energies.append(stats["outside_squared"])
    assert energies[1] < energies[0]


def test_constant_or_nonfinite_construction_is_not_fit():
    with pytest.raises(ValueError, match="variation"):
        fit_linear(np.ones((4, 3)), np.ones((4, 3)))
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_linear(np.full((4, 3), np.nan), np.ones((4, 3)))

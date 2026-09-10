#!/usr/bin/env python3
"""Direction spaces, equal-author positions and action-preservation witnesses."""

import io

import numpy as np
import pytest

from grain_research.vector_mechanisms.position_reader import (
    contractions,
    errors,
    fit_position,
    hidden_basis,
    position_design,
    predict_position,
    project_field,
)


def test_shared_rank_two_directions_recover_a_new_field():
    rng = np.random.default_rng(31)
    v = np.linalg.qr(rng.normal(size=(9, 2)))[0].T
    source = [rng.normal(size=(n, 2)) @ v / n for n in (11, 17, 23)]
    model = hidden_basis(source, rank=2)
    held = rng.normal(size=(19, 2)) @ v / 19
    np.testing.assert_allclose(project_field(held, model["basis"], 2), held, atol=1e-12)


def test_tiny_energy_mode_can_contain_all_directional_action():
    j = np.diag([100.0, 1.0, 0.0])
    p = project_field(j, np.array([[1.0, 0.0, 0.0]]), 1)
    d = np.zeros((1, 3, 3))
    d[0, 1, 1] = 1
    result = errors(j, p, d)
    assert result["field_error_ratio"] < 0.0001
    assert result["actual"][0] == 1
    assert result["predicted"][0] == 0
    assert result["error_bound"][0] == 1


def test_constant_density_has_equal_author_mass_despite_length():
    source = [np.full((10, 3), 2 / 10), np.full((100, 3), 6 / 100)]
    model = fit_position(source, 0)
    np.testing.assert_allclose(model["coefficients"], np.full((1, 3), 4))
    for count in (7, 50, 500):
        np.testing.assert_allclose(predict_position(count, model).sum(0), 4, atol=1e-13)


def test_cubic_density_transfers_between_position_grids():
    rng = np.random.default_rng(33)
    coef = rng.normal(size=(4, 5))
    source = [position_design(n, 3) @ coef / n for n in (11, 16, 23)]
    model = fit_position(source, 3)
    np.testing.assert_allclose(model["coefficients"], coef, atol=1e-12)
    np.testing.assert_allclose(
        predict_position(37, model), position_design(37, 3) @ coef / 37, atol=1e-12
    )


def test_saved_position_reader_needs_only_length_and_coefficients():
    rng = np.random.default_rng(35)
    model = fit_position([rng.normal(size=(n, 5)) for n in (13, 17, 22)], 3)
    buffer = io.BytesIO()
    np.savez(buffer, **model)
    buffer.seek(0)
    with np.load(buffer) as z:
        restored = dict(z)
    np.testing.assert_array_equal(
        predict_position(18, model), predict_position(18, restored)
    )


def test_projection_error_decreases_for_nested_hidden_subspaces():
    rng = np.random.default_rng(36)
    j = rng.normal(size=(22, 9))
    d = rng.normal(size=(4, 22, 9))
    model = hidden_basis([j], rank=4)
    ratios = [
        errors(j, project_field(j, model["basis"], k), d)["field_error_ratio"]
        for k in (1, 2, 4)
    ]
    assert np.all(np.diff(ratios) <= 0)


def test_uniform_reader_cannot_read_a_mean_null_contrast():
    d = np.array([[[-1.0, 2.0], [1.0, -2.0]]])
    assert contractions(np.ones((2, 2)), d)[0] == 0
    assert contractions(np.array([[0.0, 0.0], [1.0, 0.0]]), d)[0] == 1


def test_invalid_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        position_design(1, 3)
    with pytest.raises(ValueError):
        position_design(12, 2)
    with pytest.raises(ValueError):
        hidden_basis([np.ones((5, 3))], rank=3)
    with pytest.raises(ValueError):
        fit_position([np.ones((2, 4))], 3)
    with pytest.raises(ValueError):
        contractions(np.ones((3, 4)), np.ones((2, 4, 3)))

#!/usr/bin/env python3
"""Synthetic character, unit, source-isolation and error witnesses for HM37."""

import numpy as np
import pytest

from grain_research.construction import math_conditional_active_reading as C


def test_all_characters_invert_without_effect_selection():
    rng = np.random.default_rng(37)
    a = rng.normal(size=(8, 3, 2, 5))
    np.testing.assert_array_equal(C.characters().T @ C.characters(), np.eye(8) * 8)
    np.testing.assert_allclose(C.decode_characters(C.encode_cells(a)), a, atol=1e-14)


def test_cell_index_and_declared_mismatch():
    for i, cell in enumerate(C.CELLS):
        assert i == cell[0] * 4 + cell[1] * 2 + cell[2]
        assert C.CELLS[i ^ 3] == (cell[0], cell[1] ^ 1, cell[2] ^ 1)


def test_source_isolation():
    with pytest.raises(ValueError, match="application"):
        C.source_laws(np.ones((2, 8, 3, 2, 4)), ["source", "application"])


def test_no_condition_world_all_marginals_agree():
    a = np.ones((2, 8, 3, 2, 4)) * 2
    laws = C.source_laws(a, ["source"] * 2)["laws"]
    np.testing.assert_allclose(laws, 2)


def test_conditional_only_world_global_loses_action():
    a = np.broadcast_to(
        C.characters()[:, 1][None, :, None, None, None], (2, 8, 3, 2, 1)
    ).copy()
    laws = C.source_laws(a, ["source"] * 2)["laws"]
    _, predicted = C.predict(laws, 0, 0.5, [1, 2], [1])
    np.testing.assert_allclose(predicted[0], 0)
    np.testing.assert_allclose(predicted[3], [1, 2])
    np.testing.assert_allclose(predicted[4], [-1, -2])


def test_node_only_world_fiber_loses_motion():
    a = np.broadcast_to(
        np.array([-1.0, 0, 1])[None, None, :, None, None], (2, 8, 3, 2, 1)
    ).copy()
    laws = C.source_laws(a, ["source"] * 2)["laws"]
    _, p = C.predict(laws, 0, 1, [2, 3], [4])
    np.testing.assert_allclose(p[2], 0)
    np.testing.assert_allclose(p[1], [8, 12])


def test_unit_H_scaling_and_no_cold_covector_argument():
    laws = C.source_laws(np.ones((2, 8, 3, 2, 2)), ["source"] * 2)["laws"]
    coefficients, predicted = C.predict(laws, 3, 0, [2, 5], [3, 4])
    np.testing.assert_allclose(coefficients[:, 0], 2)
    np.testing.assert_allclose(predicted[:, 1], 35)


def test_quadratic_lagrange_rule_at_and_between_nodes():
    nodes = np.array(C.NODES)
    values = np.stack((nodes**2 + 2 * nodes + 1, 3 * nodes**2 - nodes), axis=1)
    np.testing.assert_allclose(C.interpolate(values, 0.25), [1.5625, -0.0625])
    for i, t in enumerate(C.NODES):
        np.testing.assert_array_equal(C.interpolate(values, t), values[i])


def test_signed_error_decomposition_and_cancellation():
    metric = C.error_metrics(np.array([[1.0, 3.0, 1.5], [-1.0, -3.0, -1.5]]))
    assert metric["carrier_error_rms"] == 2
    assert metric["reading_error_rms"] == 1.5
    assert metric["rmse"] == 0.5 and metric["error_cross_mean"] == -6
    assert metric["error_identity"] == 0


def test_equal_source_wordings_residual_account():
    a = np.stack((np.zeros((8, 3, 2, 2)), np.ones((8, 3, 2, 2)) * 2))
    fitted = C.source_laws(a, ["source"] * 2)
    np.testing.assert_allclose(fitted["cells"], 1)
    np.testing.assert_allclose(fitted["wording_residual"].mean(axis=0), 0)

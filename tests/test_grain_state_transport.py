#!/usr/bin/env python3
"""Small synthetic checks for actual state generation and frozen coordinates."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.state_transport import (
    fit_transport,
    pack_state,
    reconstruct,
    state_errors,
    unpack_state,
)


def test_pack_preserves_distinct_slots():
    p = np.arange(12).reshape(3, 4)
    q = np.arange(4) + 20
    pp, qq = unpack_state(pack_state(p, q), p.shape)
    np.testing.assert_array_equal(pp, p)
    np.testing.assert_array_equal(qq, q)
    with pytest.raises(ValueError):
        pack_state(p, np.zeros(5))


def test_linear_state_transport_recovers_independent_inputs():
    rng = np.random.default_rng(90)
    x = rng.normal(size=(30, 8))
    a = rng.normal(size=(8, 8)) / 5
    y = x + x @ a + 2
    model = fit_transport(x, y, alpha=1e-8)
    z = rng.normal(size=8)
    result = reconstruct(model, z)
    np.testing.assert_allclose(result["transport1"], z + z @ a + 2, atol=1e-6)
    np.testing.assert_allclose(result["residual_mean"], z + (y - x).mean(0))
    errors = state_errors(result["transport1"], z + z @ a + 2, model, (1, 4))
    assert errors["P_rmse"] < 1e-6


def test_quadratic_transport_keeps_interactions_and_batch_invariance():
    rng = np.random.default_rng(91)
    x = rng.uniform(-1, 1, size=(90, 4))
    y = x.copy()
    y[:, 0] += 3 * x[:, 0] * x[:, 1]
    y[:, 1] += x[:, 2] ** 2
    model = fit_transport(x, y, alpha=1e-8)
    z = rng.uniform(-1, 1, size=4)
    truth = z.copy()
    truth[0] += 3 * z[0] * z[1]
    truth[1] += z[2] ** 2
    before = reconstruct(model, z)
    np.testing.assert_allclose(before["transport2"], truth, atol=1e-5)
    reconstruct(model, rng.normal(size=4))
    np.testing.assert_array_equal(
        reconstruct(model, z)["transport2"], before["transport2"]
    )
    assert np.linalg.norm(before["transport2"] - truth) < np.linalg.norm(
        before["transport1"] - truth
    )


def test_degenerate_fit_and_changed_dimensions_rejected():
    with pytest.raises(ValueError):
        fit_transport(np.ones((4, 8)), np.ones((4, 8)))
    with pytest.raises(ValueError):
        fit_transport(np.eye(4), np.eye(4), alpha=0)
    model = fit_transport(np.eye(4), np.eye(4))
    with pytest.raises(ValueError):
        reconstruct(model, np.ones(5))

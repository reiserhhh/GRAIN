#!/usr/bin/env python3
"""Parity is independent of degree and imposes a paired approximation lower bound."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.configuration_parity import (
    paired_even_error,
    parity_components,
)


def test_cubic_and_quartic_function_show_odd_is_not_only_linear():
    x = np.linspace(-2, 2, 31)

    def f(z):
        return 3 + 2 * z + 4 * z * z + z**3 - 0.5 * z**4

    parts = parity_components(f(x), f(-x), np.full_like(x, 3))
    assert parts["odd"] == pytest.approx(2 * x + x**3)
    assert parts["even"] == pytest.approx(4 * x * x - 0.5 * x**4)
    assert parts["odd"] + parts["even"] == pytest.approx(parts["plus"])


def test_paired_even_error_identity_with_correlated_nonzero_parts():
    p = np.array([0.1, 0.2, -0.4, 0.3])
    n = np.array([-0.2, 0.1, -0.3, 0.1])
    h = np.array([0.2, 0.2, 0.4, 0.2])
    result = paired_even_error(p, n, h)
    assert result["paired_mse"] == pytest.approx(
        result["even_mse"] + result["odd_energy"]
    )
    assert result["paired_mse"] >= result["odd_energy"]


def test_conditioned_quadratic_is_even_for_any_fixed_context():
    rng = np.random.default_rng(19)
    d = rng.normal(size=(11, 6))
    v = rng.normal(size=(6, 4))
    z = rng.normal(size=(11, 3))
    weights = np.ones(4) + z @ rng.normal(size=(3, 4))
    assert np.sum((d @ v) ** 2 * weights) == pytest.approx(
        np.sum((-d @ v) ** 2 * weights)
    )


def test_anchor_coordinates_cast_float64_on_actual_mps_when_available():
    import torch

    from scripts.run_grain_configuration_parity import anchor_coordinates

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        state = torch.zeros((1, 6, 4), dtype=torch.float32, device=device)
        mask = np.array([False, False, True, True, False, False])
        anchored = anchor_coordinates(
            state,
            mask,
            np.ones((2, 4), dtype=np.float64),
            np.full(4, 2.0, dtype=np.float64),
        )
        assert anchored.dtype == torch.float32
        assert anchored[0, 2].cpu().numpy() == pytest.approx(np.ones(4))
        assert anchored[0, -1].cpu().numpy() == pytest.approx(np.full(4, 2.0))
        assert torch.count_nonzero(state).item() == 0

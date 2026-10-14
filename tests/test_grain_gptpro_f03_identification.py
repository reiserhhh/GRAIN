"""Exact assumptions, bounds and bootstrap endpoint checks."""

import itertools

import pytest

from grain_research.gptpro_foundation_v3.f03_identification import (
    bootstrap_endpoints,
    population_view,
)


def test_all_frozen_population_bounds_contain_truth():
    for r, pi, kappa, lam, gamma, z, design in itertools.product(
        [0.2, 0.5, 0.8],
        [0.4, 0.8],
        [-0.5, 0.5],
        [0, 0.8],
        [0, 1],
        [-1, 1],
        ["free", "common", "fixed"],
    ):
        result = population_view(r, pi, kappa, lam, gamma, z, design)
        for key in ("general_bound", "independent_opportunity_bound"):
            low, high = result[key]
            assert low - 1e-12 <= r <= high + 1e-12
        if gamma == 0 or design == "fixed":
            assert result["oracle_observed_propensity_mean"] == pytest.approx(r)
        assert result["published_only"]["f"] is None


def test_even_oracle_observed_propensity_does_not_fix_mnar():
    result = population_view(0.5, 0.8, 0.5, 0.8, 1, 1, "free")
    assert abs(result["oracle_observed_propensity_mean"] - 0.5) > 0.05


def test_constant_cell_weights_cannot_change_selected_mean():
    result = population_view(
        0.5, 0.8, 0.5, 0.8, 1, 1, "free", response_x=0, selection_x=0
    )
    assert result["oracle_observed_propensity_mean"] == pytest.approx(
        result["selected_mean"]
    )


def test_bootstrap_zero_opportunities_retains_unknown_interval():
    rows = [
        {"condition": {"z": 1}, "opportunity": 0, "selected": 0, "observation": None}
        for _ in range(8)
    ]
    result = bootstrap_endpoints(rows, 1, repetitions=100)
    assert result["zero_opportunity_draws"] == 100
    assert result["independent_low"] == [0.0, 0.0]
    assert result["independent_high"] == [1.0, 1.0]
    assert bootstrap_endpoints(rows, -1)["status"] == "NO_CONDITION_SUPPORT"

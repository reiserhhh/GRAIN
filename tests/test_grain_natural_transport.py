#!/usr/bin/env python3
"""Finite witnesses for common operators, level separation and material changes."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.natural_transport import (
    correlation,
    estimand_rows,
    fit_operator,
    paired_summary,
    removal_coordinate,
    response_channels,
    signed_energy_allocation,
)


def test_shared_operator_different_inputs_explain_changes():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(30, 3, 4))
    beta = np.array([1.0, -0.5, 0.2, 0.0])
    y = 0.3 + x @ beta
    for kind in ("between", "within", "transition"):
        fitted = fit_operator(x, y, kind, alpha=0)
        np.testing.assert_allclose(fitted["coef"], beta, atol=1e-12)
        natural = rng.normal(size=(12, 4))
        np.testing.assert_allclose(natural @ fitted["coef"], natural @ beta, atol=1e-12)


def test_estimands_do_not_confuse_author_level_and_view_response():
    levels = np.arange(9)[:, None, None]
    within = np.array([-1.0, 0, 1])[None, :, None]
    x = levels + within
    y = (levels - within)[:, :, 0]
    assert fit_operator(x, y, "between", 0)["coef"][0] == pytest.approx(1)
    assert fit_operator(x, y, "within", 0)["coef"][0] == pytest.approx(-1)
    a, b = estimand_rows(x, y, "transition")
    np.testing.assert_allclose(a, 0)
    np.testing.assert_allclose(b, 0)


def test_accounting_preserves_cancelling_cross_terms():
    native = np.array([0.01, -0.02, 0.03])
    functional = np.array([1.0, -2.0, 3.0])
    represented = np.array([-0.5, 0.2, 0.3])
    channels, audit = response_channels(native, functional, represented)
    np.testing.assert_allclose(channels.sum(1), native, atol=1e-15)
    assert audit["energy_error"] < 1e-14
    assert np.trace(audit["second_moment"]) > 100 * audit["native_energy"]


@pytest.mark.parametrize("removed", [[0], [1, 4], [0, 1, 2, 3]])
def test_removal_mixture_identity(removed):
    row = removal_coordinate(np.array([1, -0.3, 0.2, 2, -0.5]), removed)
    assert row["delta"] == pytest.approx(row["fraction"] * row["contrast"])
    assert row["identity_error"] < 1e-15


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [8], [0, 1, 2], [0.2]])
def test_invalid_removals_rejected(indices):
    with pytest.raises(ValueError):
        removal_coordinate([0.1, 0.2, 0.3], indices)


def test_constant_is_undefined_and_bootstrap_is_reproducible():
    assert correlation(np.ones(4), np.arange(4)) is None
    a = paired_summary([1, 2, 4, 5], [0.9, 3, 2, 5], draws=29)
    b = paired_summary([1, 2, 4, 5], [0.9, 3, 2, 5], draws=29)
    assert a == b


def test_fit_does_not_depend_on_subsequent_scoring_batch():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(20, 3, 2))
    fitted = fit_operator(x, x[:, :, 0], "between", 100)
    beta = np.asarray(fitted["coef"])
    point = np.array([0.2, 0.7])
    batch = np.array([point, [1000.0, -1000.0]])
    assert point @ beta == pytest.approx((batch @ beta)[0])


def test_signed_energy_allocation_keeps_cancellation():
    native = np.array([1, 2, 3.0])
    channels = np.column_stack((2 * native, -native))
    np.testing.assert_allclose(signed_energy_allocation(native, channels), [2, -1])
    with pytest.raises(ValueError):
        signed_energy_allocation(np.zeros(3), channels)


def test_part_maps_sum_to_total_map_and_anchor():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(30, 3, 5))
    parts = rng.normal(size=(30, 3, 3))
    for kind in ("between", "within", "transition"):
        fits = [fit_operator(x, parts[:, :, i], kind) for i in range(3)]
        total = fit_operator(x, 0.4 + parts.sum(2), kind)
        np.testing.assert_allclose(
            np.sum([m["coef"] for m in fits], axis=0), total["coef"], atol=1e-12
        )
        assert sum(m["intercept"] for m in fits) + (
            0.4 if kind == "between" else 0
        ) == pytest.approx(total["intercept"])

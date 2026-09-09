#!/usr/bin/env python3
"""Finite identities that keep author levels, paths and coordinates distinct."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.logic_levels import (
    coordinate_accounting,
    covariance_accounting,
    endpoint_accounting,
    infer_levels,
    level_associations,
    residual_pullback_example,
)


def opposite_panel():
    """Generate a positive between-author but negative within-author mechanism."""
    u = np.arange(9.0) - 4
    t = np.array([-1.0, 0, 1])
    x = u[:, None] + t[None, :]
    y = u[:, None] - 2 * t[None, :]
    return x[:, :, None], y, u


def test_total_covariance_exact_and_signs_differ():
    x, y, c = opposite_panel()
    account = covariance_accounting(x, y)
    assert np.max(np.abs(account["closure_error"])) < 1e-14
    links = level_associations(x, y, c)
    assert links["between"][0] == pytest.approx(1)
    assert links["within"][0] == pytest.approx(-1)
    assert account["total_r_contribution"][0] == pytest.approx(
        account["between_r_contribution"][0] + account["within_r_contribution"][0]
    )


def test_author_offsets_invisible_to_differences():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(24, 3, 2)), rng.normal(size=(24, 3))
    c = rng.normal(size=24)
    before = level_associations(x, y, c)
    after = level_associations(x + c[:, None, None] * 7, y, c)
    np.testing.assert_allclose(before["transition"], after["transition"], atol=1e-14)
    assert not np.allclose(before["between"], after["between"])


def test_author_constant_criterion_has_zero_within_covariance():
    rng = np.random.default_rng(9)
    x, c = rng.normal(size=(25, 3, 2)), rng.normal(size=25)
    account = covariance_accounting(x, np.repeat(c[:, None], 3, axis=1))
    np.testing.assert_allclose(account["within_covariance"], 0, atol=1e-14)


def test_resampling_reproducible_and_undefined_not_success():
    rng = np.random.default_rng(17)
    x, y, c = rng.normal(size=(15, 3, 2)), rng.normal(size=(15, 3)), np.arange(15)
    x[:, :, 1] = 1
    a = infer_levels(x, y, c, bootstrap=29, permutations=39, seed=4)
    b = infer_levels(x, y, c, bootstrap=29, permutations=39, seed=4)
    for name in a:
        np.testing.assert_allclose(a[name]["p"], b[name]["p"], equal_nan=True)
        assert np.isnan(a[name]["r"][1])
        assert np.isnan(a[name]["p"][1])
        assert a[name]["valid_bootstraps"][0] == 29


@pytest.mark.parametrize("orthogonal", [True, False])
def test_chart_covariance_and_penalty(orthogonal):
    rng = np.random.default_rng(99)
    x = rng.normal(size=(80, 4))
    y = x[:, 0] - x[:, 1] + rng.normal(size=80) * 0.1
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    a = q if orthogonal else q @ np.diag([0.1, 0.5, 2, 10])
    result = coordinate_accounting(x, y, a)
    assert result["transported_function_max_error"] < 1e-12
    assert result["covariant_refit_max_error"] < 1e-12
    assert result["projection_max_error"] < 1e-12
    if orthogonal:
        assert result["naive_refit_rms_difference"] < 1e-12
    else:
        assert result["naive_refit_rms_difference"] > 0.01


def test_invalid_panels_rejected():
    with pytest.raises(ValueError):
        covariance_accounting(np.zeros((4, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        covariance_accounting(np.full((4, 3, 1), np.nan), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        coordinate_accounting(np.eye(3), np.arange(3), np.zeros((3, 3)))


def test_logit_constant_shift_creates_probability_even_effect():
    logits = np.linspace(-2.0, 1, 12)
    sigmoid = lambda a: 1 / (1 + np.exp(-a))
    probability, logit = endpoint_accounting(
        sigmoid(logits - 0.2), sigmoid(logits), sigmoid(logits + 0.2)
    )
    assert probability["mean_abs_even"] > 1e-4
    assert probability["sd_odd"] > 1e-3
    assert logit["max_abs_even"] < 1e-14
    assert logit["sd_odd"] < 1e-14
    assert logit["common_odd_energy_fraction"] == pytest.approx(1.0)


def test_endpoint_invalid_probability_rejected():
    with pytest.raises(ValueError):
        endpoint_accounting([0, 0.2], [0.2, 0.3], [0.3, 0.4])


def test_common_patch_compatible_with_perfect_individual_axis():
    x = np.linspace(-2, 1, 12)
    sigmoid = lambda a: 1 / (1 + np.exp(-a))
    criterion = x.copy()
    result = endpoint_accounting(sigmoid(x - 0.2), sigmoid(x), sigmoid(x + 0.2))[1]
    assert np.corrcoef(x, criterion)[0, 1] == pytest.approx(1)
    assert result["common_odd_energy_fraction"] == pytest.approx(1)


def test_residual_is_part_of_exact_pullback():
    result = residual_pullback_example()
    assert result["sum_error"] < 1e-12
    assert result["represented_work"] == pytest.approx(1 / 3)
    assert result["residual_work"] == pytest.approx(2 / 3)
    assert result["error_if_residual_deleted"] > 0.6


def test_logic_edge_inventory_has_unique_typed_edges():
    from scripts.report_grain_logic_review import edge_inventory

    inventory = edge_inventory()
    assert len(inventory["edges"]) == 31
    assert len({e["id"] for e in inventory["edges"]}) == 31
    assert all(e["type_and_status"] and e["evidence_entry"] for e in inventory["edges"])

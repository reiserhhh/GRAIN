#!/usr/bin/env python3
"""A ratio audit must not conceal a physical error or an arbitrary bad quotient."""

import math

import pytest

from grain_research.construction.math_curvature_ratio_audit import exact_ratio_audit


def test_huge_ratio_two_ulps_not_fixed_absolute_physical_error():
    rows = [{"link": 1.0, "decoder": -0.9, "total": 1e-5}] * 16
    actual = math.nextafter(math.nextafter(1e10, math.inf), math.inf)
    a = exact_ratio_audit(rows, "link_ms_over_total_ms", actual)
    assert a["absolute_error"] < a["roundoff_bound"]
    with pytest.raises(ValueError):
        exact_ratio_audit(rows, "link_ms_over_total_ms", 1e10 + 0.01)


def test_near_zero_denominator_not_scientific_percentage():
    rows = [{"link": 1e-5, "decoder": -1e-5, "total": 1e-10}] * 8
    a = exact_ratio_audit(rows, "link_decoder_cross_over_total_ms", -2e10)
    assert not a["denominator_resolved_at_physical_1e8"]


def test_signed_cross_and_canceling_inner_sum():
    rows = [
        {"link": 1.0, "decoder": 1.0, "total": 2.0},
        {"link": 1.0, "decoder": -1.0, "total": 1.0},
    ]
    assert (
        exact_ratio_audit(rows, "link_decoder_cross_over_total_ms", 0.0)[
            "exact_binary_input_ratio"
        ]
        == 0
    )


def test_cannot_route_physical_quantity_or_zero_to_ratio_exception():
    rows = [{"link": 1.0, "decoder": 0.0, "total": 1.0}]
    with pytest.raises(ValueError):
        exact_ratio_audit(rows, "covector_max_error", 1.0)
    with pytest.raises(ValueError):
        exact_ratio_audit([{**rows[0], "total": 0.0}], "link_ms_over_total_ms", 0.0)

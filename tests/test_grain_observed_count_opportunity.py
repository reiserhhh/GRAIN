#!/usr/bin/env python3
"""Check signed rate accounts and retain a denominator-induced reversal."""

import numpy as np
import pytest

from grain_research.construction.math_observed_count_opportunity import (
    count_opportunity_account,
)


def test_rate_reversal_is_not_occurrence_reversal():
    value = count_opportunity_account([3], [2], [300], [100])
    assert value["raw_count_difference"][0] > 0
    assert value["rate_difference"][0] < 0
    assert value["occurrence_term"][0] > 0
    assert value["opportunity_term"][0] < 0
    np.testing.assert_allclose(value["reconstruction_error"], 0, atol=1e-14)


def test_equal_opportunity_and_equal_counts():
    a = count_opportunity_account([2], [1], [100], [100])
    assert a["opportunity_term"][0] == 0
    b = count_opportunity_account([2], [2], [200], [100])
    assert b["occurrence_term"][0] == 0
    assert b["rate_difference"][0] == -1


def test_all_signed_terms_and_invalid_input():
    rng = np.random.default_rng(1641)
    hi, lo = rng.integers(0, 20, size=(2, 20, 4))
    wh, wl = rng.integers(40, 300, size=(2, 20, 1))
    a = count_opportunity_account(hi, lo, wh, wl)
    b = count_opportunity_account(lo, hi, wl, wh)
    for key in ("rate_difference", "occurrence_term", "opportunity_term"):
        np.testing.assert_allclose(a[key], -b[key], atol=1e-14)
    np.testing.assert_allclose(a["reconstruction_error"], 0, atol=1e-14)
    with pytest.raises(ValueError):
        count_opportunity_account([1], [1], [0], [1])

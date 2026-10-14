#!/usr/bin/env python3
"""Synthetic checks for design-supported conditional author scoring."""

from fractions import Fraction

import numpy as np
import pytest

from grain_research.construction import math_history_resolution as M


def test_exact_alias_and_extra_event():
    p, even, odd = M.matched_populations()
    assert sum(even) == sum(odd) == 1
    assert M.exact_fingerprint(p, even, 4) == M.exact_fingerprint(p, odd, 4)
    assert M.exact_fingerprint(p, even, 5) != M.exact_fingerprint(p, odd, 5)
    for n in (1, 3, 9):
        assert M.exact_fingerprint(p, even, n, True) == M.exact_fingerprint(
            p, odd, n, True
        )


@pytest.mark.parametrize("n,power", [(4, 0), (5, 1), (6, 2)])
def test_bernstein_functional_identity(n, power):
    p = np.linspace(0, 1, 101)
    a = M.binomial_matrix(p, n)
    for k in range(5):
        w = M.compile_history_moment(n, 4, k, power)
        np.testing.assert_allclose(
            w @ a, p**power * M.binomial_matrix(p, 4)[k], atol=1e-14
        )


def test_cold_beta_profile_without_mixing_law():
    f = list(map(float, M.beta_fingerprint(2, 3, 6)))
    for k in range(5):
        result = M.cold_profile(
            {"source_trials": 6, "frequencies": f},
            {"history_trials": 4, "successes": k},
        )
        assert result["mean"] == pytest.approx((k + 2) / 9)
        assert result["variance"] == pytest.approx((k + 2) * (7 - k) / 810)


def test_identification_is_not_zero_uncertainty():
    p, even, _ = M.matched_populations()
    f = list(map(float, M.exact_fingerprint(p, even, 6)))
    result = M.cold_profile(
        {"source_trials": 6, "frequencies": f}, {"history_trials": 4, "successes": 2}
    )
    assert result["variance"] > 0
    f4 = list(map(float, M.exact_fingerprint(p, even, 4)))
    assert (
        M.cold_profile(
            {"source_trials": 4, "frequencies": f4},
            {"history_trials": 4, "successes": 2},
        )["mean"]
        is None
    )


def test_finite_bounds_keep_both_aliases():
    p, even, odd = M.matched_populations()
    a = M.binomial_matrix(p, 4)
    f = np.array(M.exact_fingerprint(p, even, 4), float)
    den = a[2]
    bounds, witnesses = M.finite_ratio_bounds(a, f, den * np.array(p, float), den)
    for mass in (even, odd):
        point = (
            den
            @ (np.array(p, float) * np.array(mass, float))
            / (den @ np.array(mass, float))
        )
        assert bounds[0] - 1e-12 <= point <= bounds[1] + 1e-12
    assert bounds[0] < 0.5 < bounds[1]
    for witness in witnesses:
        np.testing.assert_allclose(a @ witness, f, atol=1e-10)


def test_vector_alphabet_and_no_out_of_support_decode():
    alphabet = np.eye(4)
    np.testing.assert_array_equal(M.decode_vectors(alphabet[[3, 0]], alphabet), [3, 0])
    with pytest.raises(ValueError):
        M.decode_vectors(np.ones((1, 4)), alphabet)


def test_duplicates_are_not_extra_events_and_unknowns_fail():
    row = {
        "event_id": "e1",
        "condition": 1,
        "response": 0,
        "qualification": "self_actual",
    }
    assert M.count_distinct_events([row] * 10) == 1
    with pytest.raises(ValueError):
        M.count_distinct_events([row, row | {"response": 1}])
    with pytest.raises(ValueError):
        M.count_distinct_events([row | {"qualification": "quoted"}])


def test_no_truth_fields_and_no_variance_clipping():
    bundle = {"source_trials": 6, "frequencies": [0, 0, 0, 1, 0, 0, 0]}
    history = {"history_trials": 4, "successes": 2}
    assert M.cold_profile(bundle, history)["status"] == "INADMISSIBLE_EMPIRICAL_MOMENTS"
    with pytest.raises(ValueError):
        M.cold_profile(bundle | {"true_p": Fraction(1, 2)}, history)
    with pytest.raises(ValueError):
        M.cold_profile(bundle, history | {"personality": 0.5})

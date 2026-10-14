"""Full-enumeration and no-evidence checks for source-transition filtering."""

import itertools

import numpy as np

from grain_research.gptpro_foundation_v3.f08_filter import history_filter


def test_filter_equals_all_hidden_history_enumeration():
    """Marginalize all 2^5 histories rather than reuse the recursive formula."""
    transition = np.array([[0.15, 0.8], [0.35, 0.65]])
    initial = np.array([[0.1, 0.2], [0.3, 0.4]])
    poles = [0.9, 0.2, 0.7]
    qualifications = [0.8, 0, 0.4]
    reference = np.zeros((2, 2))
    for states in itertools.product((0, 1), repeat=5):
        mass = initial[states[0], states[1]]
        for t, (p, q) in enumerate(zip(poles, qualifications, strict=True), start=2):
            probability = transition[states[t - 2], states[t - 1]]
            mass *= probability if states[t] else 1 - probability
            mass *= q * (p if states[t] else 1 - p) + (1 - q) / 2
        reference[states[-2], states[-1]] += mass
    reference /= reference.sum()
    result = history_filter(poles, qualifications, transition, initial)
    np.testing.assert_allclose(result["pair"], reference, atol=1e-14)


def test_complete_oracle_collapses_to_observed_lags():
    """Fully observed histories recover their last two bits exactly."""
    result = history_filter(
        [0, 1, 1], [1, 1, 1], np.full((2, 2), 0.4), np.full((2, 2), 0.25)
    )
    np.testing.assert_array_equal(result["pair"], [[0, 0], [0, 1]])


def test_missing_is_not_zero_or_deleted_time():
    """Zero qualification ignores content but propagates state through all times."""
    transition = np.array([[0.1, 0.9], [0.3, 0.6]])
    initial = np.array([[1, 0], [0, 0]])
    first = history_filter([0, 0, 0], [0, 0, 0], transition, initial)
    second = history_filter([1, 1, 1], [0, 0, 0], transition, initial)
    np.testing.assert_allclose(first["pair"], second["pair"])
    assert not np.allclose(first["pair"], initial)
    assert first["status"] == "prior_only" and first["evidence_mass"] == 0

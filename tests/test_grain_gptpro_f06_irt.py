"""Scalar information formula, exact posterior, and event-ledger invariants."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f06_irt import (
    GRID,
    LAWS,
    PRIOR,
    IRTLedger,
    acquisition,
    choose,
    entropy,
    posterior_summary,
    posterior_update,
)


def test_eig_matches_expected_posterior_entropy():
    weights = PRIOR * np.linspace(0.1, 1, len(GRID))
    weights /= weights.sum()
    expected = []
    for action in range(3):
        p = LAWS[action] @ weights
        expected.append(
            entropy(weights)
            - p * entropy(posterior_update(weights, action, 1))
            - (1 - p) * entropy(posterior_update(weights, action, 0))
        )
    np.testing.assert_allclose(
        acquisition(weights, "expected_entropy_gain"), expected, atol=1e-14
    )


def test_fisher_information_is_expected_score_squared():
    for action in range(3):
        p = LAWS[action]
        exact = p * (1 - p) ** 2 + (1 - p) * p**2
        np.testing.assert_allclose(exact, p * (1 - p), atol=1e-14)
    assert choose(PRIOR, "fisher_map", 0) == 1
    assert choose(PRIOR, "expected_entropy_gain", 0) == 1


def test_ledger_matches_whole_likelihood_and_rejects_duplicates():
    ledger = IRTLedger()
    product = PRIOR.copy()
    for t, response in enumerate([1, 0, 1, 1, 0, 1]):
        action = t % 3
        ledger.append(str(t), action, response)
        product *= LAWS[action] if response else 1 - LAWS[action]
    product /= product.sum()
    np.testing.assert_allclose(ledger.weights, product, atol=1e-14)
    before = ledger.weights.copy()
    with pytest.raises(ValueError, match="duplicate"):
        ledger.append("0", 0, 1)
    np.testing.assert_array_equal(before, ledger.weights)


def test_unexpected_response_can_increase_entropy():
    weights = np.zeros(len(GRID))
    weights[20], weights[40] = 0.1, 0.9
    ledger = IRTLedger(weights=weights)
    before = posterior_summary(ledger.weights)
    after = ledger.append("surprise", 1, 0)
    assert after["entropy"] > before["entropy"]
    assert after["mean"] < before["mean"]


def test_surprise_does_not_guarantee_entropy_increase():
    ledger = IRTLedger()
    for t in range(10):
        ledger.append(str(t), 1, 1)
    before = posterior_summary(ledger.weights)
    after = ledger.append("surprise", 1, 0)
    assert after["entropy"] < before["entropy"]


def test_complete_observation_order_preserves_posterior():
    pairs = [(0, 1), (1, 0), (2, 1), (0, 0)]
    scores = []
    for sequence in (pairs, pairs[::-1]):
        ledger = IRTLedger()
        for t, (action, response) in enumerate(sequence):
            ledger.append(str(t), action, response)
        scores.append(ledger.weights)
    np.testing.assert_allclose(*scores, atol=1e-14)

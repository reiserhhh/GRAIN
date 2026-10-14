"""Finite active-discrimination reference for already-defined rival families."""

from __future__ import annotations

import numpy as np

from .rival_models import BETAS, CORNERS, FAMILIES, response_probability


def expected_information_gain(weights, candidates, condition):
    """Compute exact expected KL information for one binary condition."""
    weights = np.asarray(weights, dtype=float)
    probabilities = np.asarray([response_probability(f, b, condition) for f, b in candidates])
    predictive = float(weights @ probabilities)
    gain = 0.0
    for weight, probability in zip(weights, probabilities, strict=True):
        if weight <= 0:
            continue
        for outcome in (0, 1):
            likelihood = probability if outcome else 1 - probability
            mixture = predictive if outcome else 1 - predictive
            if likelihood > 0 and mixture > 0:
                gain += weight * likelihood * np.log(likelihood / mixture)
    return float(gain)


def update_posterior(weights, candidates, condition, observation):
    """Update a posterior after one revealed binary observation."""
    values = np.asarray([response_probability(f, b, condition) for f, b in candidates])
    likelihood = values if observation else 1 - values
    updated = np.asarray(weights) * likelihood
    if updated.sum() <= 0:
        return np.full(len(updated), 1 / len(updated))
    return updated / updated.sum()


def choose_action(weights, candidates, action, rng):
    """Select one predeclared action with no access to hidden family labels."""
    if action == "ado":
        gains = np.asarray([expected_information_gain(weights, candidates, corner) for corner in CORNERS])
        return CORNERS[int(np.argmax(gains))]
    if action == "random":
        return CORNERS[int(rng.integers(len(CORNERS)))]
    if action == "fixed":
        # The fixed arm is a deterministic eight-corner cycle; it has no
        # access to the posterior and therefore remains a clean comparator.
        return CORNERS[0]
    raise ValueError("unknown action")

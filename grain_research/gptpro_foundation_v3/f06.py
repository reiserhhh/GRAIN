"""Finite active observation policies and reactivity counterfactuals."""

from __future__ import annotations

import numpy as np


FAMILIES = ("linear", "quadratic", "threshold")
CONDITIONS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)


def family_probability(family: str, beta: float, condition: float) -> float:
    """Return the binary family response probability."""
    if family == "linear":
        score = beta * condition
    elif family == "quadratic":
        score = beta * (condition + 0.4 * condition**2)
    elif family == "threshold":
        score = beta * float(condition > 0)
    else:
        raise ValueError(f"unknown family {family}")
    return float(1.0 / (1.0 + np.exp(-score)))


def expected_information(weights: np.ndarray, candidates: list[tuple[str, float]], condition: float) -> float:
    """Compute exact expected KL information for a binary action."""
    probabilities = np.asarray([family_probability(family, beta, condition) for family, beta in candidates])
    predictive = float(weights @ probabilities)
    gain = 0.0
    for weight, probability in zip(weights, probabilities, strict=True):
        for outcome in (0, 1):
            likelihood = probability if outcome else 1.0 - probability
            mixture = predictive if outcome else 1.0 - predictive
            if likelihood > 0 and mixture > 0:
                gain += float(weight * likelihood * np.log(likelihood / mixture))
    return float(gain)


def update_posterior(weights: np.ndarray, candidates: list[tuple[str, float]], condition: float, observation: int) -> np.ndarray:
    """Update candidate weights after one binary response."""
    probabilities = np.asarray([family_probability(family, beta, condition) for family, beta in candidates])
    likelihood = probabilities if observation else 1.0 - probabilities
    updated = np.asarray(weights, dtype=float) * likelihood
    return updated / updated.sum() if updated.sum() > 0 else np.full(len(updated), 1.0 / len(updated))


def choose_action(weights: np.ndarray, candidates: list[tuple[str, float]], action: str, rng: np.random.Generator) -> float:
    """Choose a declared action without hidden family access."""
    if action == "ado":
        gains = [expected_information(weights, candidates, condition) for condition in CONDITIONS]
        return float(CONDITIONS[int(np.argmax(gains))])
    if action == "random":
        return float(CONDITIONS[int(rng.integers(len(CONDITIONS)))])
    if action == "fixed":
        return CONDITIONS[0]
    raise ValueError(f"unknown action {action}")


def reactive_response(base: float, condition: float, beta: float) -> float:
    """Return an observation with a declared responder-reactivity coefficient."""
    return float(base + beta * condition)

"""Exact nine-hypothesis F06 panel A; no Gaussian-state likelihood mixing."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.special import expit, xlogy

CORNERS = np.array(list(product((-1.0, 1.0), repeat=3)))
HYPOTHESES = tuple((axis, beta) for axis in range(3) for beta in (0.5, 1.0, 1.5))
LAWS = np.column_stack([expit(beta * CORNERS[:, axis]) for axis, beta in HYPOTHESES])
POLICIES = (
    "fixed_corner",
    "random",
    "coverage",
    "epistemic_variance",
    "expected_entropy_gain",
)


def lexical_maximum(values):
    """Resolve numerical posterior/acquisition ties by frozen lexical order."""
    values = np.asarray(values, dtype=float)
    return int(np.flatnonzero(np.isclose(values, values.max(), atol=1e-14, rtol=0))[0])


def entropy(probabilities):
    """Categorical entropy in nats with exact zero convention."""
    probabilities = np.asarray(probabilities, dtype=float)
    return -np.sum(xlogy(probabilities, probabilities), axis=-1)


def posterior_update(weights, action, response):
    """Exact Bernoulli Bayes update; hidden family is not an argument."""
    if response not in (0, 1) or action not in range(8):
        raise ValueError("invalid action/observation")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (9,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("invalid prior")
    likelihood = LAWS[action] if response else 1 - LAWS[action]
    updated = weights * likelihood
    return updated / updated.sum()


def acquisition_values(weights, policy):
    """Epistemic disagreement or expected posterior entropy reduction."""
    weights = np.asarray(weights, dtype=float)
    mean = LAWS @ weights
    if policy == "epistemic_variance":
        return np.sum(weights[None, :] * (LAWS - mean[:, None]) ** 2, axis=1)
    if policy == "expected_entropy_gain":
        binary = lambda p: -xlogy(p, p) - xlogy(1 - p, 1 - p)
        return binary(mean) - binary(LAWS) @ weights
    raise ValueError("not an adaptive acquisition")


def policy_action(weights, policy, step, random_action=0):
    """Choose a corner from history posterior only; ties use lexical order."""
    if policy == "fixed_corner":
        return 0
    if policy == "random":
        return int(random_action)
    if policy == "coverage":
        return step % 8
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    values = acquisition_values(weights, policy)
    # Numerically identical scores (including complementary corners) are ties.
    return lexical_maximum(values)


def run_policy(true_atom, uniforms, random_actions, policy):
    """Toy environment emits outcomes; selection/update never receive truth."""
    weights = np.full(9, 1 / 9)
    path = []
    for step, uniform in enumerate(uniforms):
        action = policy_action(weights, policy, step, random_actions[step])
        probability = LAWS[action, true_atom]
        response = int(uniform < probability)
        prior = weights.copy()
        weights = posterior_update(weights, action, response)
        path.append(
            {
                "step": step + 1,
                "action": action,
                "response": response,
                "prior": prior.tolist(),
                "posterior": weights.tolist(),
                "entropy_after_update": float(entropy(weights)),
            }
        )
    return path

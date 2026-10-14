"""Independent scalar IRT/CAT panel with exact finite-grid posteriors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit, xlogy

GRID = np.round(np.arange(-3.0, 3.001, 0.1), 1)
DIFFICULTIES = np.array([-1.0, 0.0, 1.0])
LAWS = expit(GRID[None, :] - DIFFICULTIES[:, None])
PRIOR = np.exp(-(GRID**2) / 2)
PRIOR /= PRIOR.sum()
POLICIES = ("fixed_center", "random", "coverage", "fisher_map", "expected_entropy_gain")


def entropy(weights):
    """Discrete entropy in nats, including exact zeros."""
    return -np.sum(xlogy(weights, weights), axis=-1)


def lexical_max(values):
    """Use declared increasing grid/candidate order for numerical ties."""
    values = np.asarray(values)
    return int(np.flatnonzero(np.isclose(values, values.max(), atol=1e-14, rtol=0))[0])


def posterior_update(weights, action, response):
    """Bayesian update from a single observed item, without latent access."""
    if action not in range(3) or response not in (0, 1):
        raise ValueError("invalid item or response")
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != GRID.shape
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("invalid posterior")
    likelihood = LAWS[action] if response else 1 - LAWS[action]
    posterior = weights * likelihood
    return posterior / posterior.sum()


def acquisition(weights, policy):
    """MAP Fisher is a local heuristic; EIG integrates the complete posterior."""
    if policy == "fisher_map":
        p = LAWS[:, lexical_max(weights)]
        return p * (1 - p)
    if policy == "expected_entropy_gain":
        pbar = LAWS @ weights
        binary_entropy = lambda p: -xlogy(p, p) - xlogy(1 - p, 1 - p)
        return binary_entropy(pbar) - binary_entropy(LAWS) @ weights
    raise ValueError("unknown acquisition")


def choose(weights, policy, t, random_action=0):
    """Select before the response, using only prefix posterior and fixed RNG."""
    if policy == "fixed_center":
        return 1
    if policy == "random":
        if random_action not in range(3):
            raise ValueError("invalid random item")
        return int(random_action)
    if policy == "coverage":
        return t % 3
    return lexical_max(acquisition(weights, policy))


def posterior_summary(weights):
    """Exact scalar mean, variance and equal-tail grid interval."""
    mean = float(weights @ GRID)
    cumulative = np.cumsum(weights)
    low, high = GRID[np.searchsorted(cumulative, [0.025, 0.975])]
    return {
        "mean": mean,
        "variance": float(weights @ (GRID - mean) ** 2),
        "entropy": float(entropy(weights)),
        "ci_low": float(low),
        "ci_high": float(high),
        "boundary_mass": float(weights[0] + weights[-1]),
    }


@dataclass
class IRTLedger:
    """Update once per observed event; replayed history does not add evidence."""

    weights: np.ndarray = field(default_factory=lambda: PRIOR.copy())
    seen: set[str] = field(default_factory=set)

    def append(self, event_id, action, response):
        """Reject duplicates before mutation and return the new posterior state."""
        if not isinstance(event_id, str) or not event_id or event_id in self.seen:
            raise ValueError("invalid or duplicate event")
        updated = posterior_update(self.weights, action, response)
        self.seen.add(event_id)
        self.weights = updated
        return posterior_summary(updated)

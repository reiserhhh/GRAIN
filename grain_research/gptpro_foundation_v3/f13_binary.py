#!/usr/bin/env python3
"""Finite latent-state law for a next-event probability and its binary outcome."""

import numpy as np


def next_probability_components(weights, probabilities):
    """Condition on fixed state-specific probabilities, not estimated-parameter CI.

    The interval covers the finite random probability P(state), not the unknown
    deterministic posterior mean. A learned reader/filter defines a working law;
    this function does not make its weights empirically calibrated.
    """
    w, p = np.asarray(weights, float).ravel(), np.asarray(probabilities, float).ravel()
    if (
        w.shape != p.shape
        or not len(w)
        or not np.isfinite(w).all()
        or not np.isfinite(p).all()
        or (w < 0).any()
        or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0)
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("normalized state weights and bounded probabilities required")
    mean = float(w @ p)
    state = float(w @ ((p - mean) ** 2))
    outcome = float(w @ (p * (1 - p)))
    order = np.argsort(p)
    cumulative = np.cumsum(w[order])
    low = p[order[min(np.searchsorted(cumulative, 0.025), len(p) - 1)]]
    high = p[order[min(np.searchsorted(cumulative, 0.975), len(p) - 1)]]
    return {
        "mean": mean,
        "state_probability_variance": state,
        "conditional_outcome_variance": outcome,
        "total_outcome_variance": state + outcome,
        "probability_lower": float(low),
        "probability_upper": float(high),
    }

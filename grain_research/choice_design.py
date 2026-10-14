#!/usr/bin/env python3
"""Opportunity-offset choice, conditional cell scores and learned observation design.

The ingredients are standard binomial likelihood, method-of-moments empirical
Bayes and linear minimum mean-square prediction. Text evidence and the declared
reference conditions determine their local measurement meaning.
"""

from __future__ import annotations

import numpy as np


def choice_coordinates(root, tail, opportunity_log_odds=None):
    """Score a two-level choice tree, excluding externally assigned observations.

    Inputs contain only freely chosen -1/+1 roots and tails. With known opportunity
    log odds the output estimates preference offsets; otherwise it is an observed
    choice profile. The 0.5 pseudocount regularizes finite counts, not absent nodes.
    """
    root, tail = np.asarray(root), np.asarray(tail)
    if root.ndim != 1 or root.shape != tail.shape:
        raise ValueError("Aligned one-dimensional choice arrays required")
    if not np.isin(root, [-1, 1]).all() or not np.isin(tail, [-1, 1]).all():
        raise ValueError("Choice codes must be -1/+1")
    offset = (
        np.zeros(3)
        if opportunity_log_odds is None
        else np.asarray(opportunity_log_odds, float)
    )
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Three finite opportunity offsets required")
    branches = (root, tail[root == -1], tail[root == 1])
    counts = np.array([len(x) for x in branches])
    score = np.full(3, np.nan)
    for index, values in enumerate(branches):
        if len(values):
            positive = (values == 1).sum()
            score[index] = np.log((positive + 0.5) / (len(values) - positive + 0.5))
            score[index] -= offset[index]
    return score, counts


def anchored_cells(condition, evidence):
    """Return equally referenced level/contrast; no score for an absent cell.

    Level=(m_minus+m_plus)/2 and response=(m_plus-m_minus)/2 are contrasts
    of two actual conditions, not extrapolation to an unobserved middle condition.
    """
    condition, evidence = np.asarray(condition), np.asarray(evidence, float)
    if condition.ndim != 1 or condition.shape != evidence.shape:
        raise ValueError("Aligned one-dimensional evidence required")
    if not np.isin(condition, [-1, 1]).all() or not np.isfinite(evidence).all():
        raise ValueError("Finite evidence at -1/+1 conditions required")
    counts = np.array([(condition == z).sum() for z in (-1, 1)])
    means = np.array(
        [
            evidence[condition == z].mean() if n else np.nan
            for z, n in zip((-1, 1), counts)
        ]
    )
    return np.array([means.mean(), (means[1] - means[0]) / 2]), counts


def learn_cell_moments(pilot, eigen_floor=1e-6):
    """Learn source mean, covariance and event variance without latent parameters.

    Pilot shape is author x two conditions x independent events. Sampling variance
    of each author's cell mean is removed from the between-author covariance. A
    recorded eigenvalue floor provides a positive working covariance, not new data.
    """
    y = np.asarray(pilot, float)
    if y.ndim != 3 or y.shape[1] != 2 or min(y.shape[0], y.shape[2]) < 3:
        raise ValueError("At least three source authors/events and two cells required")
    if not np.isfinite(y).all() or eigen_floor <= 0:
        raise ValueError("Finite pilot and positive eigenvalue floor required")
    means = y.mean(axis=2)
    noise = y.var(axis=2, ddof=1).mean(axis=0)
    if np.any(noise <= 0):
        raise ValueError("Pilot must identify positive within-cell variance")
    uncorrected = np.cov(means, rowvar=False, ddof=1)
    corrected = uncorrected - np.diag(noise / y.shape[2])
    eigenvalues, vectors = np.linalg.eigh(corrected)
    covariance = (vectors * np.maximum(eigenvalues, eigen_floor)) @ vectors.T
    return {
        "mean": means.mean(axis=0),
        "covariance": covariance,
        "noise": noise,
        "raw_corrected_eigenvalues": eigenvalues,
        "floor_count": int((eigenvalues < eigen_floor).sum()),
        "source_authors": y.shape[0],
        "events_per_cell": y.shape[2],
    }


def linear_cell_score(condition, evidence, event_ids, moments):
    """Predict two cell expectations from new events and fixed source moments.

    An exact repeated report contributes once. Conflicting values under the same
    event ID require an explicit revision/measurement-error model and are rejected.
    Returned observed and borrowed terms sum to the prediction; neither is concealed.
    """
    z, y, ids = (
        np.asarray(condition),
        np.asarray(evidence, float),
        np.asarray(event_ids),
    )
    if z.ndim != 1 or z.shape != y.shape or z.shape != ids.shape:
        raise ValueError("Aligned event observations required")
    if not np.isin(z, [-1, 1]).all() or not np.isfinite(y).all():
        raise ValueError("Finite two-condition evidence required")
    seen = {}
    for cell, value, event in zip(z, y, ids):
        key = str(event)
        if key in seen and seen[key] != (cell, value):
            raise ValueError(
                "Conflicting event reports require explicit revision handling"
            )
        seen[key] = (cell, value)
    counts, sums = np.zeros(2), np.zeros(2)
    for cell, value in seen.values():
        index = int(cell == 1)
        counts[index] += 1
        sums[index] += value
    covariance = np.asarray(moments["covariance"], float)
    noise = np.asarray(moments["noise"], float)
    mean = np.asarray(moments["mean"], float)
    if covariance.shape != (2, 2) or noise.shape != (2,) or mean.shape != (2,):
        raise ValueError("Two-cell source moments required")
    if not all(np.isfinite(v).all() for v in (covariance, noise, mean)):
        raise ValueError("Finite source moments required")
    if (
        not np.allclose(covariance, covariance.T)
        or np.linalg.eigvalsh(covariance).min() <= 0
    ):
        raise ValueError("Positive definite source covariance required")
    if np.any(noise <= 0):
        raise ValueError("Positive event noise required")
    prior_precision = np.linalg.inv(covariance)
    posterior = np.linalg.inv(prior_precision + np.diag(counts / noise))
    observed = posterior @ (sums / noise)
    borrowed = posterior @ prior_precision @ mean
    return {
        "score": observed + borrowed,
        "observed": observed,
        "borrowed": borrowed,
        "data_weights": posterior @ np.diag(counts / noise),
        "prior_weights": posterior @ prior_precision,
        "covariance": posterior,
        "counts": counts,
        "reports": len(z),
        "unique_events": len(seen),
    }


def design_gain(covariance, target, noise, *, repeated=False):
    """Predict target MSE reduction from one new event in each condition.

    The value is conditional on the source second-moment model. Perfect repetition
    has zero independent information. Reader bias/domain shift is not modeled away.
    """
    covariance, target, noise = map(np.asarray, (covariance, target, noise))
    if covariance.shape != (2, 2) or target.shape != (2,) or noise.shape != (2,):
        raise ValueError("Two-cell covariance, target and noise required")
    if not all(np.isfinite(v).all() for v in (covariance, target, noise)) or np.any(
        noise <= 0
    ):
        raise ValueError("Finite moments and positive noise required")
    if repeated:
        return np.zeros(2)
    return (target @ covariance) ** 2 / (noise + np.diag(covariance))


def choose_allocation(moments, target, budget, warm_per_cell=2):
    """Choose a fixed-budget source-informed design without target-author outcomes."""
    if budget < 2 * warm_per_cell or warm_per_cell < 0:
        raise ValueError("Budget must cover the nonnegative warm start")
    counts = np.full(2, warm_per_cell, dtype=int)
    history = []
    precision = np.linalg.inv(moments["covariance"])
    for _ in range(budget - counts.sum()):
        covariance = np.linalg.inv(precision + np.diag(counts / moments["noise"]))
        gain = design_gain(covariance, np.asarray(target), moments["noise"])
        index = int(np.argmax(gain))
        history.append(
            {"counts_before": counts.tolist(), "gains": gain.tolist(), "chosen": index}
        )
        counts[index] += 1
    return counts, history

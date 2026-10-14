#!/usr/bin/env python3
"""Conditional author evidence from fixed vector menus, not personality truth."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


def count(value, name):
    """Accept exact nonnegative trial counts, not rounded continuous readings."""
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
        raise ValueError(f"{name}: finite integer count required")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name}: nonnegative integer count required")
    return result


def log_choose(n, x):
    """Return log binomial multiplicity on its declared feasible support."""
    return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)


def conditional_law(n, r, total, contrast):
    """Condition two independent binomials on their sum; remove shared intercept.

    contrast is a declared log-odds difference, not a reader probability or a
    psychological score. Its support includes degenerate all-zero/all-one totals.
    """
    n, r, total = (count(v, k) for v, k in zip((n, r, total), ("n", "r", "K")))
    if total > n + r or not np.isfinite(contrast):
        raise ValueError("Feasible total and finite log-odds contrast required")
    support = np.arange(max(0, total - r), min(n, total) + 1, dtype=float)
    log_weight = log_choose(n, support) + log_choose(r, total - support)
    log_weight += support * contrast
    log_mass = log_weight - logsumexp(log_weight)
    mass = np.exp(log_mass)
    mean = float(mass @ support)
    variance = float(mass @ ((support - mean) ** 2))
    return support, log_mass, mean, variance


def geometry(menus, tolerance=1e-10):
    """Identify menu contrasts and their exact finite coefficient equivalences."""
    d = np.asarray(menus, float)
    if d.ndim != 2 or min(d.shape) < 1 or not np.isfinite(d).all():
        raise ValueError("Nonempty finite menu difference matrix required")
    if not 0 < tolerance < 1:
        raise ValueError("Relative rank tolerance in (0,1) required")
    _, singular, vh = np.linalg.svd(d, full_matrices=True)
    rank = int(np.sum(singular > tolerance * singular[0])) if singular[0] else 0
    basis = vh[:rank].T
    null = vh[rank:].T
    return {
        "rank": rank,
        "dimensions": d.shape[1],
        "singular": singular,
        "basis": basis,
        "null": null,
        "projector": basis @ basis.T,
        "null_error": float(np.max(abs(d @ null))) if null.size else 0.0,
    }


def cold_score(records, menus, parameter_grid):
    """Score only qualified choice counts relative to fixed reference counts.

    No truth, participant identity, text, questionnaire, teacher output or menu
    inference enters this function. Forced reports remain outside this choice
    likelihood. Unknown qualification is rejected instead of becoming a zero.
    """
    d, grid = np.asarray(menus, float), np.asarray(parameter_grid, float)
    if d.ndim != 2 or grid.ndim != 2 or d.shape[1] != grid.shape[1]:
        raise ValueError("Aligned menu/grid dimensions required")
    if len(grid) < 2 or not np.isfinite(d).all() or not np.isfinite(grid).all():
        raise ValueError("Finite menus and at least two fixed prior atoms required")
    allowed = {"menu", "n", "success", "ref_n", "ref_success", "qualification"}
    likelihood = np.zeros(len(grid))
    used, forced, informative, seen = 0, 0, 0, set()
    for record in records:
        if set(record) != allowed:
            raise ValueError("Cold input has missing or unauthorized fields")
        j = count(record["menu"], "menu")
        if j >= len(d) or j in seen:
            raise ValueError("Unknown or duplicated menu address")
        seen.add(j)
        n, x, r, y = (
            count(record[k], k) for k in ("n", "success", "ref_n", "ref_success")
        )
        if x > n or y > r:
            raise ValueError("Success count exceeds trials")
        qualification = record["qualification"]
        if qualification == "forced":
            forced += n
            continue
        if qualification != "free":
            raise ValueError("Unknown event/offer qualification cannot be scored")
        if not n or not r:
            continue
        used += 1
        support, _, _, _ = conditional_law(n, r, x + y, 0.0)
        if len(support) == 1:
            continue
        informative += 1
        coefficients = log_choose(n, support) + log_choose(r, x + y - support)
        action = grid @ d[j]
        normalizer = logsumexp(
            coefficients[:, None] + support[:, None] * action, axis=0
        )
        likelihood += log_choose(n, x) + log_choose(r, y) + x * action - normalizer
    posterior = np.exp(likelihood - logsumexp(likelihood))
    mean = posterior @ grid
    centered = grid - mean
    covariance = (centered.T * posterior) @ centered
    information = np.zeros((d.shape[1], d.shape[1]))
    for record in records:
        if record["qualification"] != "free" or not record["n"] or not record["ref_n"]:
            continue
        j = count(record["menu"], "menu")
        _, _, _, variance = conditional_law(
            record["n"],
            record["ref_n"],
            record["success"] + record["ref_success"],
            d[j] @ mean,
        )
        information += variance * np.outer(d[j], d[j])
    return {
        "status": "OBSERVED_CONDITIONAL_CHOICE"
        if informative
        else "NO_IDENTIFYING_CHOICE",
        "posterior": posterior,
        "mean": mean,
        "covariance": covariance,
        "information": information,
        "log_likelihood": likelihood,
        "menus_used": used,
        "informative_menus": informative,
        "forced_trials_excluded": forced,
        "menu_contrasts": d @ mean,
    }

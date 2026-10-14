"""Exact F03 population bounds and resampling of observable endpoint estimates."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def population_view(
    r, pi, kappa, lam, gamma, z, design, response_x=0.15, selection_x=0.8
):
    """Enumerate a conditional population, including oracle propensity adjustment.

    The oracle is an algebraic diagnostic of selection assumptions, not a fitted
    application estimator. Independence of opportunity and response is explicit.
    """
    opportunity = pi if design == "free" else 1.0
    f = g = weighted_y = weighted_mass = 0.0
    for x in (0, 1):
        p = r + response_x * (2 * x - 1)
        probabilities = [1 - p, p]
        q = [
            1.0
            if design == "fixed"
            else float(
                expit(kappa + lam * z + selection_x * (2 * x - 1) + gamma * (2 * y - 1))
            )
            for y in (0, 1)
        ]
        marginal_q = sum(probabilities[y] * q[y] for y in (0, 1))
        for y in (0, 1):
            selected_mass = 0.5 * probabilities[y] * q[y]
            f += opportunity * selected_mass
            g += opportunity * selected_mass * y
            weighted_mass += selected_mass / marginal_q
            weighted_y += selected_mass * y / marginal_q
    return {
        "target": r,
        "pi": opportunity,
        "f": f,
        "g": g,
        "selected_mean": g / f,
        "general_bound": [g, g + 1 - f],
        "independent_opportunity_bound": [
            g / opportunity,
            g / opportunity + 1 - f / opportunity,
        ],
        "oracle_observed_propensity_mean": weighted_y / weighted_mass,
        "published_only": {
            "f": None,
            "response_bound": [0.0, 1.0],
            "reason": "no opportunity denominator",
        },
    }


def bootstrap_endpoints(rows, z, seed=20260915, repetitions=2000):
    """Percentile CIs for plug-in endpoints from condition-specific opportunity rows.

    Resample the observable four-cell multinomial: O=0; O=1/S=0;
    O=S=1/Y=0; O=S=1/Y=1. These are endpoint-estimation intervals, not
    automatically a simultaneous confidence region for the latent response.
    """
    cell = [row for row in rows if row["condition"]["z"] == z]
    if not cell:
        return {"status": "NO_CONDITION_SUPPORT"}
    counts = np.zeros(4, dtype=int)
    for row in cell:
        index = (
            0
            if not row["opportunity"]
            else 1
            if not row["selected"]
            else 2 + int(row["observation"])
        )
        counts[index] += 1
    draws = np.random.default_rng(seed).multinomial(
        len(cell), counts / len(cell), size=repetitions
    )
    n_o = draws[:, 1:].sum(axis=1)
    n_s = draws[:, 2:].sum(axis=1)
    general_low = draws[:, 3] / len(cell)
    general_high = general_low + 1 - n_s / len(cell)
    positive = n_o > 0
    independent_low = np.divide(
        draws[:, 3], n_o, out=np.zeros(repetitions), where=positive
    )
    independent_high = np.divide(
        draws[:, 3] + n_o - n_s, n_o, out=np.ones(repetitions), where=positive
    )
    return {
        "status": "ENDPOINT_PERCENTILE_NOT_LATENT_TARGET_CI",
        "repetitions": repetitions,
        "condition_n": len(cell),
        "zero_opportunity_draws": int(np.sum(~positive)),
        **{
            name: np.quantile(values, [0.025, 0.975]).tolist()
            for name, values in [
                ("general_low", general_low),
                ("general_high", general_high),
                ("independent_low", independent_low),
                ("independent_high", independent_high),
            ]
        },
    }

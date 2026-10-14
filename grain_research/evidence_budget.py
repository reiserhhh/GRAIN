#!/usr/bin/env python3
"""Detection-availability models, separate from personality scoring or validity."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import betabinom, hypergeom


def fit_beta_binomial(counts, observations):
    """Fit heterogeneous match rates on source authors, using likelihood only."""
    counts = np.asarray(counts)
    if observations <= 0 or np.any(counts < 0) or np.any(counts > observations):
        raise ValueError("Counts outside exposure range")
    if not np.equal(counts, np.floor(counts)).all() or len(counts) < 2:
        raise ValueError("At least two integer counts required")
    p = (counts.sum() + 0.5) / (len(counts) * observations + 1)
    fits = []
    for concentration in (1.0, 10.0, 100.0):
        start = np.log([p * concentration, (1 - p) * concentration])
        fit = minimize(
            lambda q: -betabinom.logpmf(counts, observations, *np.exp(q)).sum(),
            start,
            method="L-BFGS-B",
            bounds=[(-12, 12), (-12, 12)],
        )
        if fit.success and np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise RuntimeError("Source beta-binomial MLE did not converge")
    best = min(fits, key=lambda r: r.fun)
    alpha, beta = np.exp(best.x)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "nll": float(best.fun),
        "successful_starts": len(fits),
        "boundary": bool(np.any(np.abs(best.x) > 11.9)),
    }


def beta_coverage(budget, alpha, beta):
    """Predict at least one detector hit under a beta-mixture of comment rates."""
    if budget < 0 or alpha <= 0 or beta <= 0:
        raise ValueError("Invalid budget or beta parameters")
    return float(-np.expm1(betaln(alpha, beta + budget) - betaln(alpha, beta)))


def finite_coverage(counts, observations, budget):
    """Exact without-replacement coverage within each observed author's comment pool."""
    counts = np.asarray(counts)
    if (
        not 0 <= budget <= observations
        or np.any(counts < 0)
        or np.any(counts > observations)
    ):
        raise ValueError("Invalid finite sampling counts or budget")
    return hypergeom.sf(0, observations, counts, budget)

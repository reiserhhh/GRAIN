#!/usr/bin/env python3
"""Finite-inner-sample bias accounting for nested covariance components."""

import numpy as np


def inner_sampling_correction(personal, source, outer, inner):
    """Unbiased component estimates conditional on fixed source fits and iid draws.

    Inputs use ddof0 at both levels. Corrected source covariance may be indefinite;
    do not floor it or confuse it with a usable posterior covariance. These are
    Monte Carlo component diagnostics, not new frozen prediction intervals.
    """
    p, s = np.asarray(personal, float), np.asarray(source, float)
    if (
        outer < 2
        or inner < 2
        or p.shape != s.shape
        or not np.isfinite(p).all()
        or not np.isfinite(s).all()
    ):
        raise ValueError("finite aligned components with B,J >=2 required")
    inner_mean_variance = p / (inner - 1)
    return {
        "personal_corrected": p * inner / (inner - 1),
        "source_corrected": s - (1 - 1 / outer) * inner_mean_variance,
        "source_mc_contamination": (1 - 1 / outer) * inner_mean_variance,
        "total_correction": inner_mean_variance / outer,
    }

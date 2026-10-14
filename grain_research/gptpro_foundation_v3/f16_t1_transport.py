#!/usr/bin/env python3
"""Exact source/pilot-product uncertainty for the fixed-support T1 target."""

import numpy as np

from .f16_t1_source import CONDITIONS, CONTRAST, verify_bundle


def source_kernels(bundle, conditions):
    """Compile Gaussian source components to four cell-mean action kernels."""
    verify_bundle(bundle)
    z = np.asarray(conditions, float)
    if z.ndim != 1 or not np.isfinite(z).all() or not np.isin(z, CONDITIONS).all():
        raise ValueError("declared four-condition observation schedule required")
    counts = np.array([(z == c).sum() for c in CONDITIONS], float)
    if np.any(counts == 0):
        raise ValueError("transport kernel requires complete condition support")
    offsets, weights, variances = [], [], []
    for model in bundle["source_models"]:
        mu = np.asarray(model["mu"])
        cov = np.asarray(model["Sigma"])
        total = cov + np.diag(model["sigma2"] / counts)
        w = np.linalg.solve(total, cov @ CONTRAST)
        variance = float(CONTRAST @ cov @ CONTRAST - w @ total @ w)
        if variance < 0 or not np.isfinite(variance):
            raise ValueError("invalid conditional variance")
        offsets.append(float(CONTRAST @ mu - w @ mu))
        weights.append(w)
        variances.append(variance)
    return {
        "offsets": np.array(offsets),
        "weights": np.array(weights),
        "conditional": np.array(variances),
        "conditions": z,
    }


def cell_means(values, conditions):
    """Preserve all within-condition readings, allowing pilot and author axes."""
    y, z = np.asarray(values, float), np.asarray(conditions, float)
    if y.shape[-1] != len(z) or not np.isfinite(y).all():
        raise ValueError("finite observations and aligned schedule required")
    if any(not np.any(z == c) for c in CONDITIONS):
        raise ValueError("all four conditions required")
    return np.stack([y[..., z == c].mean(-1) for c in CONDITIONS], axis=-1)


def integrate(kernels, means):
    """Integrate every source by pilot pair, not paired bootstrap indices."""
    m = np.asarray(means, float)
    if m.ndim != 3 or m.shape[-1] != 4 or not np.isfinite(m).all() or not len(m):
        raise ValueError("pilotdraw by author by four-cell array required")
    # Dimension order: pilot resample, application author, source resample.
    pieces = np.einsum("jac,sc->jas", m, kernels["weights"]) + kernels["offsets"]
    conditional = np.full(m.shape[1], kernels["conditional"].mean())
    source = pieces.var(-1, ddof=0).mean(0)
    pilot = pieces.mean(-1).var(0, ddof=0)
    score = pieces.mean((0, 2))
    variance = conditional + source + pilot
    return {
        "score": score,
        "variance": variance,
        "conditional": conditional,
        "source": source,
        "pilot": pilot,
        "lower": score - 1.96 * np.sqrt(variance),
        "upper": score + 1.96 * np.sqrt(variance),
    }


def bootstrap_indices(seed, number=200, authors=60):
    """Draw whole pilot authors independently of source and application outcomes."""
    return np.random.default_rng(np.random.SeedSequence([seed, 163, 12])).integers(
        0, authors, (number, authors)
    )

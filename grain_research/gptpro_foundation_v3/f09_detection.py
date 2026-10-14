"""Composite-null future checks with source-only parametric calibration."""

import numpy as np
from scipy.special import logsumexp, xlogy

from .f09_rivals import EVEN, ODD, candidates, probability


def count_predictions(development):
    """Exact nine-atom posterior from three observations per development corner."""
    counts = np.asarray(development, dtype=float)
    if counts.shape[-1] != 4 or not np.isin(counts, [0, 1, 2, 3]).all():
        raise ValueError("four binomial counts with three trials required")
    atoms = candidates()["atoms"]
    source = np.array([probability(EVEN, f, b) for f, b in atoms])
    future = np.array([probability(ODD, f, b) for f, b in atoms])
    logweight = counts @ np.log(source).T + (3 - counts) @ np.log1p(-source).T
    weights = np.exp(logweight - logsumexp(logweight, axis=-1, keepdims=True))
    return weights @ future


def discrepancy(counts, prediction):
    """Four-cell binomial deviance, with 0 log 0 defined by continuity."""
    counts = np.asarray(counts, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if (
        counts.shape != prediction.shape
        or counts.shape[-1] != 4
        or not np.isin(counts, [0, 1, 2, 3]).all()
    ):
        raise ValueError("aligned binomial future counts required")
    if (
        not np.isfinite(prediction).all()
        or ((prediction <= 0) | (prediction >= 1)).any()
    ):
        raise ValueError("interior predictive probabilities required")
    return 2 * np.sum(
        xlogy(counts, counts / (3 * prediction))
        + xlogy(3 - counts, (3 - counts) / (3 * (1 - prediction))),
        axis=-1,
    )


def calibrate(seed, repetitions=2000):
    """Simulate development and future independently for each declared null atom."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9, 901]))
    dev = []
    future = []
    statistics = []
    for family, beta in candidates()["atoms"]:
        d = rng.binomial(3, probability(EVEN, family, beta), size=(repetitions, 4))
        f = rng.binomial(3, probability(ODD, family, beta), size=(repetitions, 4))
        dev.append(d)
        future.append(f)
        statistics.append(discrepancy(f, count_predictions(d)))
    return {
        "development": np.array(dev),
        "future": np.array(future),
        "statistics": np.array(statistics),
    }


def score_oof(development, future, calibration):
    """Conservative maximum survival p across null atoms, never pooled quantiles."""
    statistic = float(discrepancy(np.asarray(future), count_predictions(development)))
    values = np.asarray(calibration)
    if values.ndim != 2 or values.shape[0] != 9 or not np.isfinite(values).all():
        raise ValueError("nine finite calibration distributions required")
    p = (1 + np.sum(values >= statistic, axis=1)) / (values.shape[1] + 1)
    return {
        "statistic": statistic,
        "p_by_atom": p,
        "p_composite": float(p.max()),
        "rejected": bool(p.max() < 0.05),
    }

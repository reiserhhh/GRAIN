"""F04 finite-prefix variance and tail estimators, without latent inputs."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from grain_research.gptpro_foundation_v3.f04_recovery import observed_arrays


def predict_tail(prefix, *, mode="linear", interval="gaussian", draws=2000, seed=0):
    """Fit prefix OLS and return variance inference and operational tail counts.

    The fixed-design bootstrap resamples centered leverage-adjusted residuals,
    refits every draw, and uses a basic interval for the residual variance.
    Gaussian chi-square coverage requires a correctly specified mean model.
    Neither the latent variance nor future outcomes enter this function.
    """
    if interval not in {"gaussian", "bootstrap"} or draws < 2:
        raise ValueError("invalid interval or resampling budget")
    x, y = observed_arrays(prefix, mode)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    rank = int(np.linalg.matrix_rank(x))
    df = len(y) - rank
    if df <= 0:
        raise ValueError("no residual degrees of freedom")
    residual = y - x @ beta
    rss = float(residual @ residual)
    variance = rss / df
    projection = x @ np.linalg.pinv(x)
    leverage = np.diag(projection)
    adjusted = residual / np.sqrt(np.maximum(1 - leverage, 1e-12))
    adjusted -= adjusted.mean()
    if interval == "gaussian":
        low, high = rss / chi2.ppf([0.975, 0.025], df)
    else:
        rng = np.random.default_rng(seed)
        simulated = rng.choice(adjusted, size=(draws, len(y)), replace=True)
        # (I-H) refits each fixed-design OLS sample without a solver loop.
        refit_residual = simulated @ (np.eye(len(y)) - projection)
        estimates = np.sum(refit_residual**2, axis=1) / df
        qlow, qhigh = np.quantile(estimates, [0.025, 0.975])
        low, high = max(0.0, 2 * variance - qhigh), max(0.0, 2 * variance - qlow)
    threshold = 2 * np.sqrt(variance)
    return {
        "beta": beta.tolist(),
        "rank": rank,
        "df": df,
        "rss": rss,
        "variance": variance,
        "variance_ci_low": float(low),
        "variance_ci_high": float(high),
        "interval": interval,
        "bootstrap_draws": draws if interval == "bootstrap" else 0,
        "operational_threshold": float(threshold),
        "prefix_operational_tail": float(np.mean(np.abs(residual) > threshold)),
        "prefix_leverage_adjusted_tail": float(np.mean(np.abs(adjusted) > threshold)),
    }

#!/usr/bin/env python3
"""Independent-pilot variance estimation and explicit observation-unit transport.

REML, Gaussian conditioning and total variance are inherited methods. This module
keeps population learning, paired reader calibration and new-person scoring apart.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from grain_research.temporal_measurement import posterior, score_target, target_vector


def calibration(reference, observed, condition, *, conditional=False):
    """Fit a paired SAME-UNIT meter map, not a mapping of population quantiles.

    The reference score is the chosen observation unit, not psychological truth.
    Conditional maps require the supplied condition to be observed at deployment.
    """
    a, b, z = (np.asarray(x, float) for x in (reference, observed, condition))
    if a.ndim != 1 or a.shape != b.shape or a.shape != z.shape or len(a) < 8:
        raise ValueError("At least eight aligned paired anchor observations required")
    if not all(np.isfinite(v).all() for v in (a, b, z)):
        raise ValueError("Finite anchors required")
    x = (
        np.c_[np.ones(len(a)), a, z, z * a]
        if conditional
        else np.c_[np.ones(len(a)), a]
    )
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Paired calibration design is rank deficient")
    coef = np.linalg.lstsq(x, b, rcond=None)[0]
    error = b - x @ coef
    return {
        "coef": coef,
        "conditional": conditional,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "n": len(a),
    }


def restore_units(observed, condition, fitted):
    """Invert the learned meter with positive gain; do not recenter new authors."""
    y, z = np.asarray(observed, float), np.asarray(condition, float)
    coef = np.asarray(fitted["coef"])
    offset, gain = coef[:2]
    if fitted["conditional"]:
        offset, gain = offset + coef[2] * z, gain + coef[3] * z
    if np.any(np.asarray(gain) <= 1e-6):
        raise ValueError("Meter orientation or support invalid at requested condition")
    return (y - offset) / gain


def covariance_parameters(theta):
    """Positive profile covariance, OU stationary variance and decay rate."""
    v = np.asarray(theta, float)
    if v.shape != (8,):
        raise ValueError("Eight covariance parameters required")
    l = np.array(
        [[np.exp(v[0]), 0, 0], [v[1], np.exp(v[2]), 0], [v[3], v[4], np.exp(v[5])]]
    )
    return l @ l.T, float(np.exp(v[6])), float(np.exp(v[7])), l


def reml_objective(theta, evidence, design, times, noise, *, gradient=True):
    """Profiled REML for independent authors on a common opportunity schedule.

    y_u ~ N(X*m, X*V*X' + p*exp(-rate*|dt|) + noise*I).
    Population mean is profiled by GLS; the fixed-effect determinant is retained.
    No individual latent parameters or evaluation outcomes are supplied.
    """
    y, x, t = map(np.asarray, (evidence, design, times))
    n, count = y.shape
    v, p, rate, l = covariance_parameters(theta)
    distance = np.abs(t[:, None] - t[None, :])
    corr = np.exp(-rate * distance)
    k = x @ v @ x.T + p * corr + noise * np.eye(count)
    factor = cho_factor(k, lower=True)
    ki = cho_solve(factor, np.eye(count))
    kix = ki @ x
    fixed = x.T @ kix
    ff = cho_factor(fixed, lower=True)
    mean = cho_solve(ff, x.T @ ki @ y.mean(axis=0))
    residual = y - (x @ mean)[None, :]
    scatter = residual.T @ residual
    logdet = 2 * np.log(np.diag(factor[0])).sum()
    fixed_logdet = 2 * np.log(np.diag(ff[0])).sum() + 3 * np.log(n)
    value = 0.5 * (n * logdet + np.sum(ki * scatter.T) + fixed_logdet)
    if not gradient:
        return float(value), mean, k
    sensitivity = 0.5 * (n * ki - ki @ scatter @ ki - kix @ cho_solve(ff, kix.T))
    derivatives = []
    for i, j in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)):
        dl = np.zeros((3, 3))
        dl[i, j] = l[i, j] if i == j else 1.0
        derivatives.append(x @ (dl @ l.T + l @ dl.T) @ x.T)
    derivatives.extend((p * corr, -p * rate * distance * corr))
    grad = np.array([np.sum(sensitivity * dk.T) for dk in derivatives])
    return float(value), grad


def fit_pilot(replicates, condition, partner, times, *, maxiter=160):
    """Learn profile/state moments from independent pilot authors and replicates.

    Replicates share the current latent occasion, identifying within-occasion
    observation error. Independent event reports can meet that design in the
    simulation; arbitrary repeated sentences cannot be declared replicates.
    """
    y = np.asarray(replicates, float)
    z, u, t = (np.asarray(v, float) for v in (condition, partner, times))
    if y.ndim != 3 or y.shape[0] < 8 or y.shape[2] < 2 or y.shape[1] < 8:
        raise ValueError("Eight pilot authors/occasions and two replicates required")
    if z.shape != (y.shape[1],) or u.shape != z.shape or t.shape != z.shape:
        raise ValueError("Aligned common schedule required")
    if not all(np.isfinite(v).all() for v in (y, z, u, t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Finite observations and strictly increasing times required")
    x = np.c_[np.ones(len(t)), z, u]
    if np.linalg.matrix_rank(x) < 3:
        raise ValueError(
            "Pilot must identify separate level, condition and partner columns"
        )
    means = y.mean(axis=2)
    noise = float(y.var(axis=2, ddof=1).mean() / y.shape[2])
    if noise <= 1e-12:
        raise ValueError("Independent replicate noise is not identified")
    beta = np.linalg.lstsq(x, means.T, rcond=None)[0].T
    corrected = np.cov(beta, rowvar=False) - noise * np.linalg.inv(x.T @ x)
    ev, vectors = np.linalg.eigh(corrected)
    v0 = (vectors * np.maximum(ev, 1e-5)) @ vectors.T
    l = np.linalg.cholesky(v0)
    residual = means - beta @ x.T
    p0 = max(float(np.var(residual)) - noise, 0.002)
    theta0 = np.array(
        [
            np.log(l[0, 0]),
            l[1, 0],
            np.log(l[1, 1]),
            l[2, 0],
            l[2, 1],
            np.log(l[2, 2]),
            np.log(p0),
            np.log(np.log(2) / (2 * np.median(np.diff(t)))),
        ]
    )
    bounds = [
        (-10, 1),
        (-2, 2),
        (-10, 1),
        (-2, 2),
        (-2, 2),
        (-10, 1),
        (-12, 2),
        (-8, 4),
    ]
    opt = minimize(
        reml_objective,
        theta0,
        args=(means, x, t, noise),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )
    v, p, rate, _ = covariance_parameters(opt.x)
    _, mean, _ = reml_objective(opt.x, means, x, t, noise, gradient=False)
    return {
        "mean": mean,
        "covariance": v,
        "stationary": p,
        "rate": rate,
        "noise": noise,
        "theta": opt.x,
        "success": bool(opt.success),
        "message": str(opt.message),
        "iterations": int(opt.nit),
        "objective": float(opt.fun),
        "gradient_max": float(np.max(np.abs(opt.jac))),
        "initial_floor_count": int((ev < 1e-5).sum()),
        "bound_count": sum(
            abs(a - b) < 1e-5 or abs(a - c) < 1e-5 for a, (b, c) in zip(opt.x, bounds)
        ),
        "authors": len(y),
        "replicates": y.shape[2],
    }


def score_batch(evidence, condition, partner, times, fitted, targets):
    """Apply one frozen pilot model to all new authors, without refitting."""
    y = np.asarray(evidence, float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Finite author-by-occasion evidence required")
    z, u = np.asarray(condition), np.asarray(partner)
    model = posterior(
        times,
        np.zeros((len(times), 1)),
        z[:, None],
        u,
        drift=np.array([[-fitted["rate"]]]),
        stationary=np.array([[fitted["stationary"]]]),
        noise=np.array([[fitted["noise"]]]),
        profile_mean=fitted["mean"],
        profile_cov=fitted["covariance"],
    )
    predictions, variances = [], []
    for kind in targets:
        target = target_vector(
            model,
            kind,
            0,
            z=0.5 if kind == "conditional_profile" else z[-1],
            partner=0.0 if kind == "conditional_profile" else u[-1],
        )
        result = score_target(model, target)
        predictions.append(result["offset"] + y @ result["kernel"])
        variances.append(result["variance"])
    return np.stack(predictions, axis=1), np.broadcast_to(
        variances, (len(y), len(targets))
    ).copy()


def total_uncertainty(predictions, variances):
    """Approximate pilot-bootstrap mixture, not an exact Bayesian posterior.

    Preserve conditional observation variance and between-construction variation
    separately. Coverage is an experimental output, never guaranteed by this sum.
    """
    p, v = np.asarray(predictions, float), np.asarray(variances, float)
    if (
        p.shape != v.shape
        or p.ndim < 2
        or len(p) < 2
        or not np.isfinite(p).all()
        or not np.isfinite(v).all()
        or np.any(v < 0)
    ):
        raise ValueError(
            "At least two aligned finite nonnegative-variance constructions required"
        )
    within, between = v.mean(axis=0), p.var(axis=0, ddof=1)
    return {
        "score": p.mean(axis=0),
        "within": within,
        "between": between,
        "variance": within + between,
    }

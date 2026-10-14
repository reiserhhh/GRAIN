"""Source-learned conditional moments and target-directed observation design.

These are linear second-moment predictors, not exact posteriors for binary text
events. Realized application values are absent from the design interface.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from grain_research.pilot_transport import covariance_parameters, reml_objective

LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def hidden_basis(z):
    """A function exactly absent on all three original source conditions."""
    z = np.asarray(z, float)
    return np.where(np.isin(z, [-1, 0, 1]), 0.0, np.sin(np.pi * z))


def basis(z, kind="quadratic"):
    """Map observed opportunity coordinates, not author identities, to columns."""
    z = np.atleast_1d(np.asarray(z, float))
    if not np.isfinite(z).all() or np.any(np.abs(z) > 1):
        raise ValueError("Finite declared condition support [-1,1] required")
    if kind == "cells":
        result = (z[:, None] == LEVELS[None, :]).astype(float)
        if not np.all(result.sum(axis=1) == 1):
            raise ValueError("Cell model has no support for this new condition")
        return result
    x = np.c_[np.ones(len(z)), z, z**2]
    if kind == "hidden":
        return np.c_[x, hidden_basis(z)]
    if kind != "quadratic":
        raise ValueError("Unknown response basis")
    return x


def psd_floor(v, floor=1e-8):
    """Record the finite-sample moment repair instead of hiding negative roots."""
    e, q = np.linalg.eigh((v + v.T) / 2)
    return (q * np.maximum(e, floor)) @ q.T, int((e < floor).sum()), e


def replicate_noise(y, z):
    """Estimate one-report error using independent same-occasion replicates."""
    y, z = np.asarray(y, float), np.asarray(z, float)
    if y.ndim != 3 or y.shape[2] < 2 or len(z) != y.shape[1]:
        raise ValueError("Author x occasion x independent replicates required")
    if not np.isfinite(y).all() or y.shape[0] < 8:
        raise ValueError("Eight finite source authors required")
    v = y.var(axis=2, ddof=1)
    levels = np.unique(z)
    noise = np.array([v[:, z == c].mean() for c in levels])
    if np.any(noise <= 1e-10):
        raise ValueError("Positive independent report variance required")
    return levels, noise


def fit_source(y, z, times, maxiter=200):
    """Fit quadratic person moments and correlated state using source text only."""
    y, z, t = np.asarray(y, float), np.asarray(z, float), np.asarray(times, float)
    levels, noise = replicate_noise(y, z)
    if not np.array_equal(levels, [-1, 0, 1]) or np.any(np.diff(t) <= 0):
        raise ValueError("Three original source levels and increasing times required")
    x = basis(z)
    means = y.mean(axis=2)
    noise_at = np.array([noise[np.flatnonzero(levels == c)[0]] for c in z])
    # One-report variance, divided only for the source replicate MEAN.
    r = noise_at / y.shape[2]
    a = np.linalg.pinv(x)
    beta = means @ a.T
    v0, floors, _ = psd_floor(np.cov(beta, rowvar=False) - (a * r) @ a.T, 1e-5)
    l = np.linalg.cholesky(v0)
    residual = means - beta @ x.T
    s0 = max(float(np.var(residual) - np.mean(r)), 0.002)
    theta = np.array(
        [
            np.log(l[0, 0]),
            l[1, 0],
            np.log(l[1, 1]),
            l[2, 0],
            l[2, 1],
            np.log(l[2, 2]),
            np.log(s0),
            np.log(np.log(2) / 2),
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
        theta,
        args=(means, x, t, r),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )
    v, s, rate, _ = covariance_parameters(opt.x)
    _, mean, _ = reml_objective(opt.x, means, x, t, r, gradient=False)
    log_noise_coef = np.linalg.solve(basis(levels), np.log(noise))
    return dict(
        kind="quadratic",
        mean=mean,
        covariance=v,
        stationary=s,
        rate=rate,
        noise_levels=levels,
        noise=noise,
        log_noise_coef=log_noise_coef,
        success=bool(opt.success),
        message=str(opt.message),
        iterations=int(opt.nit),
        gradient_max=float(np.abs(opt.jac).max()),
        initial_floor_count=floors,
        bound_count=sum(
            min(abs(a - b), abs(a - c)) < 1e-5 for a, (b, c) in zip(opt.x, bounds)
        ),
        authors=len(y),
        replicates=y.shape[2],
        theta=opt.x,
    )


def fit_expanded(y, z, times, original):
    """Learn five-cell moments from ADDITIONAL independent source authors.

    The old source state kernel is inherited. No hidden basis or true parameter
    is supplied; a finite-cell map cannot then claim still-unseen support.
    """
    y, z, t = np.asarray(y, float), np.asarray(z, float), np.asarray(times, float)
    levels, noise = replicate_noise(y, z)
    if not np.array_equal(levels, LEVELS):
        raise ValueError("Expanded source must cover all five cells")
    x = basis(z, "cells")
    a = np.linalg.pinv(x)
    beta = y.mean(axis=2) @ a.T
    r = np.array([noise[np.flatnonzero(levels == c)[0]] for c in z]) / y.shape[2]
    nuisance = original["stationary"] * np.exp(
        -original["rate"] * np.abs(t[:, None] - t[None, :])
    ) + np.diag(r)
    v, floors, eigenvalues = psd_floor(np.cov(beta, rowvar=False) - a @ nuisance @ a.T)
    return dict(
        kind="cells",
        mean=beta.mean(axis=0),
        covariance=v,
        stationary=original["stationary"],
        rate=original["rate"],
        noise_levels=levels,
        noise=noise,
        floor_count=floors,
        raw_eigenvalues=eigenvalues,
        authors=len(y),
        replicates=y.shape[2],
    )


def report_noise(model, z):
    """Application variance is for ONE report, not a source replicate mean."""
    z = np.asarray(z, float)
    if model["kind"] == "quadratic":
        return np.exp(basis(z) @ np.asarray(model["log_noise_coef"]))
    levels, noise = np.asarray(model["noise_levels"]), np.asarray(model["noise"])
    return np.array([noise[np.flatnonzero(levels == c)[0]] for c in z])


def target_vector(model, target):
    """Declare profile, finite contrast or last-occasion total response target."""
    plus, minus = basis([0.5, -0.5], model["kind"])
    if target == "new_condition_profile":
        return plus, False
    if target == "new_condition_contrast":
        return (plus - minus) / 2, False
    if target == "current_response":
        return plus, True
    raise ValueError("Unknown target")


def moments(model, z, times, event_ids, target, *, naive_duplicates=False):
    """Observation/target moments; exact aliases share their observation error."""
    z, t, ids = np.asarray(z, float), np.asarray(times, float), np.asarray(event_ids)
    if z.shape != t.shape or ids.shape != t.shape:
        raise ValueError("Aligned condition, time and event IDs required")
    for identity in np.unique(ids):
        mask = ids == identity
        if np.unique(z[mask]).size != 1 or np.unique(t[mask]).size != 1:
            raise ValueError("Same event cannot have conflicting condition or time")
    x = basis(z, model["kind"])
    v, mean = np.asarray(model["covariance"]), np.asarray(model["mean"])
    q, state = target_vector(model, target)
    r = report_noise(model, z)
    same = np.eye(len(ids)) if naive_duplicates else (ids[:, None] == ids[None, :])
    kernel = x @ v @ x.T + model["stationary"] * np.exp(
        -model["rate"] * np.abs(t[:, None] - t[None, :])
    )
    kernel += same * np.sqrt(r[:, None] * r[None, :])
    cross = x @ v @ q
    variance = float(q @ v @ q)
    if state:
        cross += model["stationary"] * np.exp(-model["rate"] * np.abs(t - 19))
        variance += model["stationary"]
    return dict(
        K=kernel,
        k=cross,
        variance=variance,
        obs_mean=x @ mean,
        target_mean=float(q @ mean),
    )


def score_weights(m, selected):
    """Freeze a linear score from moments without observing application values."""
    s = np.asarray(selected, int)
    if not len(s):
        return np.zeros(0), float(m["target_mean"]), float(m["variance"])
    k = np.asarray(m["k"])[s]
    kernel = np.asarray(m["K"])[np.ix_(s, s)]
    # Exact duplicate observations give a singular kernel; the pseudoinverse is
    # its redundant-coordinate representation, not independent new information.
    w = np.linalg.pinv(kernel, hermitian=True, rcond=1e-12) @ k
    variance = float(m["variance"] - k @ w)
    if variance < -1e-7:
        raise ValueError("Negative target conditional variance")
    return (
        w,
        float(m["target_mean"] - np.asarray(m["obs_mean"])[s] @ w),
        max(0.0, variance),
    )


def gains(m, selected):
    """Schur-complement variance reduction for every available next observation."""
    k, kernel = np.asarray(m["k"]), np.asarray(m["K"])
    if len(selected):
        s = np.asarray(selected, int)
        inv = np.linalg.pinv(kernel[np.ix_(s, s)], hermitian=True, rcond=1e-12)
        cross = k - k[s] @ inv @ kernel[s, :]
        var = np.diag(kernel) - np.einsum("ij,ji->i", kernel[:, s] @ inv, kernel[s, :])
    else:
        cross, var = k.copy(), np.diag(kernel).copy()
    result = np.divide(cross**2, var, out=np.zeros(len(k)), where=var > 1e-10)
    result[np.asarray(selected, int)] = 0
    return result


def choose(m, z, times, ids, policy, budget, seed):
    """Target-directed, coverage or random design, before application scoring."""
    z, t, ids = np.asarray(z), np.asarray(times), np.asarray(ids)
    rng = np.random.default_rng(seed)
    selected, rows = [], []
    if budget > len(np.unique(ids)):
        raise ValueError("Budget exceeds distinct available reports")
    for step in range(budget):
        eligible = [
            j
            for j in range(len(z))
            if j not in selected
            and (policy == "source_repeat_naive" or ids[j] not in ids[selected])
        ]
        gs = gains(m, selected)
        if policy == "random":
            pick = int(rng.choice(eligible))
        elif policy == "coverage":
            # Least observed condition, then maximum temporal separation, then
            # latest event on ties. This baseline does not read the target.
            def key(j):
                count = sum(z[k] == z[j] for k in selected)
                distance = min((abs(t[j] - t[k]) for k in selected), default=20)
                return (-count, distance, t[j], -j)

            pick = max(eligible, key=key)
        else:
            pick = max(eligible, key=lambda j: (gs[j], -j))
        rows.append(
            dict(
                step=step + 1,
                candidate=pick,
                event_id=int(ids[pick]),
                condition=float(z[pick]),
                time=float(t[pick]),
                predicted_gain=float(gs[pick]),
            )
        )
        selected.append(pick)
    return selected, rows


def true_risk(truth_moments, selected, weights, offset):
    """Population MSE of a FROZEN score, including its source-estimated offset."""
    s, w = np.asarray(selected, int), np.asarray(weights, float)
    k, kernel = np.asarray(truth_moments["k"]), np.asarray(truth_moments["K"])
    bias = (
        offset
        + np.asarray(truth_moments["obs_mean"])[s] @ w
        - truth_moments["target_mean"]
    )
    return float(
        truth_moments["variance"]
        - 2 * w @ k[s]
        + w @ kernel[np.ix_(s, s)] @ w
        + bias**2
    )

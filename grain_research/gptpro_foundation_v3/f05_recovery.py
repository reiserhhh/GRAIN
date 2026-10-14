"""Observed-only dynamics estimation and correctly paired F05 simulation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped, opaque_author


def generate_dynamics(world, seed, counts=None, events=64):
    """Generate independent role episodes with genuinely shared innovations.

    The shared-shock world has common and own innovation variance .125 each.
    Initial states have stationary marginal variance; their cross-author
    covariance includes the same historical common innovation process.
    """
    worlds = ("core", "near_root", "shared_shock")
    if world not in worlds or events < 1:
        raise ValueError("unknown world or event budget")
    counts = counts or {"source": 300, "validation": 100, "application": 200}
    rows, truths = [], []
    w = worlds.index(world)
    for role_id, role in enumerate(("source", "validation", "application")):
        size = counts[role]
        if size == 0:
            continue
        parameters = []
        for i in range(size):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, 5, w, role_id, i, 0])
            )
            parameters.append(
                [
                    rng.normal(),
                    rng.normal(0.5, 0.5),
                    rng.uniform(0.9, 0.98)
                    if world == "near_root"
                    else rng.uniform(0.2, 0.8),
                ]
            )
        parameters = np.asarray(parameters)
        rho = parameters[:, 2]
        common_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 5, w, role_id, 1])
        )
        common = (
            common_rng.normal(0, np.sqrt(0.125), events)
            if world == "shared_shock"
            else np.zeros(events)
        )
        if world == "shared_shock":
            # Exact cross-author stationary initial covariance for common AR input.
            covariance = 0.125 / (1 - rho[:, None] * rho[None, :])
            covariance += np.diag(0.125 / (1 - rho**2))
        else:
            covariance = np.diag(0.25 / (1 - rho**2))
        initial_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 5, w, role_id, 2])
        )
        initial = np.linalg.cholesky(covariance) @ initial_rng.normal(size=size)
        for i, (a, b, r) in enumerate(parameters):
            author = opaque_author(seed, world, role, i)
            noise_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 5, w, role_id, i, 3])
            )
            obs_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 5, w, role_id, i, 4])
            )
            own = noise_rng.normal(
                0, np.sqrt(0.125) if world == "shared_shock" else 0.5, events
            )
            measurement = obs_rng.normal(0, 0.5, events)
            state = initial[i]
            states = []
            for t in range(events):
                state = r * state + common[t] + own[t]
                states.append(float(state))
                z = -1.0 if (t // 4) % 2 == 0 else 1.0
                rows.append(
                    {
                        "author_id": author,
                        "role": role,
                        "world": world,
                        "event_id": f"{author}:{t}",
                        "episode_id": f"{seed}:{world}:{role}",
                        "t": t,
                        "z": z,
                        "y": float(a + b * z + state + measurement[t]),
                    }
                )
            truths.append(
                {
                    "author_id": author,
                    "role": role,
                    "world": world,
                    "a": float(a),
                    "b": float(b),
                    "rho": float(r),
                    "sigma_w": 0.5,
                    "sigma_e": 0.5,
                    "initial_state": float(initial[i]),
                    "states": states,
                    "innovations": (common + own).tolist(),
                    "common_innovations": common.tolist(),
                }
            )
    return rows, truths


def arrays(rows):
    """Validate one ordered observation sequence, without accepting truth."""
    if not rows or len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate sequence")
    t = np.array([r["t"] for r in rows], dtype=int)
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must increase")
    y = np.array([r["y"] for r in rows], dtype=float)
    z = np.array([r["z"] for r in rows], dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError("nonfinite observation")
    return y, z, t


def filter_arrays(psi, y, z, times, log_sigma):
    """Kalman likelihood with exact elapsed-time transition and stationary start."""
    a, b, eta = psi
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(log_sigma))
    stationary = q / (1 - rho * rho)
    mean, variance, loglik = 0.0, stationary, 0.0
    means, variances = [], []
    previous = times[0]
    for index, (observation, condition, time) in enumerate(zip(y, z, times)):
        if index:
            gap = int(time - previous)
            transition = rho**gap
            mean *= transition
            variance = transition**2 * variance + stationary * (1 - transition**2)
        innovation = observation - a - b * condition - mean
        total = variance + r
        loglik -= 0.5 * (np.log(2 * np.pi * total) + innovation**2 / total)
        gain = variance / total
        mean += gain * innovation
        variance *= 1 - gain
        means.append(float(mean))
        variances.append(float(variance))
        previous = time
    return {"loglik": float(loglik), "state_mean": means, "state_variance": variances}


def kalman_loglik(psi, rows, log_sigma):
    """Observed-only filter wrapper."""
    return filter_arrays(psi, *arrays(rows), log_sigma)


def dense_posterior(psi, rows, log_sigma):
    """Independent dense likelihood and terminal-state Gaussian conditioning."""
    y, z, times = arrays(rows)
    a, b, eta = psi
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(log_sigma))
    variance = q / (1 - rho * rho)
    latent = variance * rho ** np.abs(times[:, None] - times[None, :])
    covariance = latent + np.eye(len(y)) * r
    residual = y - a - b * z
    chol = np.linalg.cholesky(covariance)
    logdet = 2 * np.log(np.diag(chol)).sum()
    cross = latent[-1]
    return {
        "loglik": float(
            -0.5
            * (
                len(y) * np.log(2 * np.pi)
                + logdet
                + residual @ np.linalg.solve(covariance, residual)
            )
        ),
        "state_mean": float(cross @ np.linalg.solve(covariance, residual)),
        "state_variance": float(variance - cross @ np.linalg.solve(covariance, cross)),
    }


def fit_person(rows, log_sigma, prior=None):
    """Two fixed starts; select converged ML/MAP objective on the prefix only."""
    y, z, times = arrays(rows)
    beta = np.linalg.lstsq(np.column_stack((np.ones(len(z)), z)), y, rcond=None)[0]
    precision = np.linalg.inv(prior["Sigma"]) if prior else None

    def objective(psi):
        value = -filter_arrays(psi, y, z, times, log_sigma)["loglik"]
        if prior:
            delta = psi - np.asarray(prior["mu"])
            value += 0.5 * delta @ precision @ delta
        return value

    attempts = []
    candidates = []
    for start_rho in (0.2, 0.8):
        initial = [float(y.mean()), float(beta[1]), eta_from_rho(start_rho)]
        fit = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-6},
        )
        attempts.append(
            {
                "start_rho": start_rho,
                "success": bool(fit.success),
                "objective": float(fit.fun),
                "iterations": int(fit.nit),
                "message": str(fit.message),
            }
        )
        if fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all():
            candidates.append(fit)
    if candidates:
        best = min(candidates, key=lambda f: f.fun)
        return {"psi": best.x.tolist(), "success": True, "attempts": attempts}
    fallback = prior["mu"] if prior else [0.0, 0.5, eta_from_rho(0.5)]
    return {
        "psi": list(fallback),
        "success": False,
        "attempts": attempts,
        "fallback": "source_prior" if prior else "declared_initial_source_baseline",
    }


def fit_source_dynamics(rows):
    """Two source rounds on first 48 events; freeze noise and population prior."""
    if any(row["role"] != "source" for row in rows):
        raise ValueError("source rows only")
    people = {author: events[:48] for author, events in grouped(rows).items()}
    if len(people) < 2:
        raise ValueError("at least two source authors")
    prior = None
    log_sigma = np.log([0.5, 0.5])
    history = []
    for round_id in range(2):
        fits = {
            author: fit_person(person, log_sigma, prior)
            for author, person in people.items()
        }
        cached = [
            (np.array(fits[a]["psi"]), arrays(person)) for a, person in people.items()
        ]

        def objective(logs, cached=cached):
            return -sum(
                filter_arrays(psi, *observed, logs)["loglik"]
                for psi, observed in cached
            )

        noise_fit = minimize(
            objective,
            log_sigma,
            method="L-BFGS-B",
            bounds=[(np.log(0.05), np.log(5))] * 2,
            options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-6},
        )
        if noise_fit.success and np.isfinite(noise_fit.fun):
            log_sigma = noise_fit.x
        successful = np.array([f["psi"] for f in fits.values() if f["success"]])
        if len(successful) >= 2:
            prior = {
                "mu": successful.mean(axis=0).tolist(),
                "Sigma": (np.cov(successful, rowvar=False) + np.eye(3) * 1e-4).tolist(),
            }
        elif prior is None:
            prior = {
                "mu": [0.0, 0.5, eta_from_rho(0.5)],
                "Sigma": np.diag([1.0, 0.25, 1.0]).tolist(),
            }
        history.append(
            {
                "round": round_id + 1,
                "personal_fits": fits,
                "successful_authors": len(successful),
                "total_authors": len(people),
                "noise_success": bool(noise_fit.success),
                "noise_message": str(noise_fit.message),
                "noise_objective": float(noise_fit.fun),
                "log_sigma": log_sigma.tolist(),
                "prior": prior,
            }
        )
    return {
        "log_sigma": log_sigma.tolist(),
        "prior": prior,
        "rounds": history,
        "method": "two_round_source_ML_then_MAP",
        "source_authors": len(people),
        "source_prefix_limit": 48,
        "source_events_used": sum(len(person) for person in people.values()),
    }


def score_prefix(rows, source_model, future_conditions=(), fit_personal=True):
    """Return plug-in state and future response using past observations only.

    Future conditions are a supplied design, never future outcomes. Forecast
    variance excludes parameter estimation uncertainty, reserved for F13.
    """
    prior = source_model["prior"]
    logs = np.asarray(source_model["log_sigma"])
    fit = (
        fit_person(rows, logs, prior)
        if fit_personal
        else {"psi": prior["mu"], "success": True, "attempts": []}
    )
    filtered = kalman_loglik(fit["psi"], rows, logs)
    a, b, eta = fit["psi"]
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * logs)
    state = filtered["state_mean"][-1]
    variance = filtered["state_variance"][-1]
    forecasts = []
    for k, z in enumerate(future_conditions, start=1):
        multiplier = rho**k
        forecasts.append(
            {
                "k": k,
                "condition": float(z),
                "mean": float(a + b * z + multiplier * state),
                "variance": float(
                    multiplier**2 * variance
                    + q * sum(rho ** (2 * j) for j in range(k))
                    + r
                ),
            }
        )
    return {
        "psi": fit["psi"],
        "rho": float(rho),
        "fit_success": fit["success"],
        "attempts": fit["attempts"],
        "state_mean": state,
        "state_variance": variance,
        "forecasts": forecasts,
        "variance_scope": "plug_in_not_parameter_uncertainty",
    }

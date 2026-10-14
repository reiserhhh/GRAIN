"""F06 causal timing and observed/latent-state policy likelihoods."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr


def observed_arrays(rows):
    """Reject duplicate events and truth-bearing input before state inference."""
    allowed = {"author_id", "event_id", "role", "t", "z", "y", "h"}
    if not rows or any(set(row) - allowed for row in rows):
        raise ValueError("observed event fields only")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate event")
    if len({row["author_id"] for row in rows}) != 1:
        raise ValueError("single author required")
    times = np.asarray([row["t"] for row in rows])
    if not np.array_equal(times, np.arange(len(rows))):
        raise ValueError("complete ordered prefix starting at zero required")
    y, z, h = (
        np.asarray([row[k] for row in rows], dtype=float) for k in ("y", "z", "h")
    )
    if not all(np.isfinite(v).all() for v in (y, z, h)) or not np.isin(h, [0, 1]).all():
        raise ValueError("invalid observations or actions")
    return y, z, h


def gaussian_policy_filter(rows, a, b, rho, beta, q=0.25, r=0.25, policy="exogenous"):
    """Condition on recorded actions; action at t only enters state at t+1."""
    y, z, h = observed_arrays(rows)
    if not abs(rho) < 1 or q <= 0 or r <= 0:
        raise ValueError("invalid dynamics")
    if policy == "recorded" and not np.array_equal(h, y > 0):
        return {"status": "IMPOSSIBLE_RECORDED_POLICY", "loglik": float("-inf")}
    if policy not in ("exogenous", "recorded"):
        raise ValueError("hidden-state policies require joint likelihood")
    m, v, loglik = 0.0, q / (1 - rho**2), 0.0
    means, variances = [], []
    for t, (obs, condition) in enumerate(zip(y, z, strict=True)):
        if t:
            m = rho * m + beta * h[t - 1]
            v = rho**2 * v + q
        innovation = obs - a - b * condition - m
        total = v + r
        loglik -= 0.5 * (np.log(2 * np.pi * total) + innovation**2 / total)
        m += v / total * innovation
        v *= r / total
        means.append(float(m))
        variances.append(float(v))
    return {
        "status": "OK",
        "loglik": float(loglik),
        "state_mean": means,
        "state_variance": variances,
    }


def fit_reactive_prefix(rows, source):
    """Freeze source noise/prior, extend beta prior, fit observed prefix only.

    This working fit conditions on h and deliberately omits hidden policy
    selection. Its a/b estimates are held fixed in conditional sensitivity
    profiles, not presented as an unbiased hidden-policy estimator.
    """
    y, z, _ = observed_arrays(rows)
    q, r = np.exp(2 * np.asarray(source["log_sigma"]))
    mu = np.r_[source["prior"]["mu"], 0.4]
    covariance = np.zeros((4, 4))
    covariance[:3, :3] = source["prior"]["Sigma"]
    covariance[3, 3] = 0.2**2
    precision = np.linalg.inv(covariance)

    def objective(psi):
        a, b, eta, beta = psi
        value = gaussian_policy_filter(rows, a, b, 0.99 * np.tanh(eta), beta, q, r)[
            "loglik"
        ]
        delta = psi - mu
        return -value + 0.5 * delta @ precision @ delta

    slope = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), y, rcond=None)[0][1]
    attempts, candidates = [], []
    for start in (0.2, 0.8):
        fit = minimize(
            objective,
            [y.mean(), slope, np.arctanh(start / 0.99), 0.4],
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-6},
        )
        attempts.append(
            {
                "start_rho": start,
                "success": bool(fit.success),
                "objective": float(fit.fun),
                "message": str(fit.message),
            }
        )
        if fit.success and np.isfinite(fit.fun):
            candidates.append(fit)
    chosen = min(candidates, key=lambda f: f.fun).x if candidates else mu
    return {
        "success": bool(candidates),
        "psi": chosen.tolist(),
        "attempts": attempts,
        "fallback": None if candidates else "source_prior",
        "method": "working_action_conditioned_MAP_not_hidden_policy_correction",
    }


def grid_policy_filter(
    rows, a, b, rho, beta, gammas=(0.0, 1.0, 3.0), q=0.25, r=0.25, limit=8.0, points=401
):
    """Joint p(y,h) filtering with explicit omitted tail mass before normalization.

    Parallel columns use the same state kernel for the frozen gamma grid.
    A None gamma omits policy selection (recorded/exogenous design). Grid
    quadrature is numerical, not an exact Gaussian identity. No grid expansion
    happens here; the wrapper permits exactly one declared expansion.
    """
    y, z, h = observed_arrays(rows)
    if not abs(rho) < 1 or q <= 0 or r <= 0 or points < 3 or limit <= 0:
        raise ValueError("invalid grid/dynamics")
    grid = np.linspace(-limit, limit, points)
    weights = np.full(points, 2 * limit / (points - 1))
    weights[[0, -1]] *= 0.5
    stationary = q / (1 - rho**2)
    initial_tail = float(2 * ndtr(-limit / np.sqrt(stationary)))
    if initial_tail > 0.001:
        return {
            "status": "GRID_INADEQUATE",
            "max_outside_mass": initial_tail,
            "failure_t": 0,
            "limit": limit,
            "points": points,
        }
    prior = (
        np.exp(-0.5 * grid**2 / stationary) / np.sqrt(2 * np.pi * stationary) * weights
    )
    mass = np.tile(prior[:, None], (1, len(gammas)))
    mass /= mass.sum(axis=0)
    transitions, tails = {}, {}
    for action in (0.0, 1.0):
        centers = rho * grid + beta * action
        transitions[action] = (
            np.exp(-0.5 * (grid[:, None] - centers[None, :]) ** 2 / q)
            / np.sqrt(2 * np.pi * q)
            * weights[:, None]
        )
        tails[action] = ndtr((-limit - centers) / np.sqrt(q)) + ndtr(
            (centers - limit) / np.sqrt(q)
        )
    loglik = np.zeros(len(gammas))
    means, variances, before_means, diagnostics = [], [], [], []
    largest_tail = initial_tail
    largest_quadrature_error = 0.0
    for t, (obs, condition) in enumerate(zip(y, z, strict=True)):
        if t:
            outside = tails[h[t - 1]] @ mass
            largest_tail = max(largest_tail, float(outside.max()))
            if outside.max() > 0.001:
                return {
                    "status": "GRID_INADEQUATE",
                    "max_outside_mass": largest_tail,
                    "failure_t": t,
                    "limit": limit,
                    "points": points,
                }
            predicted = transitions[h[t - 1]] @ mass
            largest_quadrature_error = max(
                largest_quadrature_error,
                float(np.max(np.abs(predicted.sum(axis=0) + outside - 1))),
            )
            mass = predicted
        else:
            outside = np.full(len(gammas), initial_tail)
        log_emission = -0.5 * (
            (obs - a - b * condition - grid) ** 2 / r + np.log(2 * np.pi * r)
        )
        offset = float(log_emission.max())
        observed = mass * np.exp(log_emission - offset)[:, None]
        before = observed / observed.sum(axis=0)
        before_means.append((grid @ before).tolist())
        selection = np.column_stack(
            [
                np.ones(points)
                if gamma is None
                else expit((2 * h[t] - 1) * gamma * grid)
                for gamma in gammas
            ]
        )
        joint = observed * selection
        norm = joint.sum(axis=0)
        if np.any(norm <= 0) or not np.isfinite(norm).all():
            return {
                "status": "NUMERIC_FAILURE",
                "failure_t": t,
                "limit": limit,
                "points": points,
            }
        loglik += np.log(norm) + offset
        mass = joint / norm
        mean = grid @ mass
        means.append(mean.tolist())
        variances.append(np.sum((grid[:, None] - mean) ** 2 * mass, axis=0).tolist())
        diagnostics.append({"t": t, "outside_mass": outside.tolist()})
    return {
        "status": "OK",
        "loglik": loglik.tolist(),
        "state_mean": means,
        "state_variance": variances,
        "before_action_state_mean": before_means,
        "gammas": list(gammas),
        "max_outside_mass": largest_tail,
        "max_quadrature_mass_error": largest_quadrature_error,
        "tail_diagnostics": diagnostics,
        "limit": limit,
        "points": points,
    }


def policy_sensitivity(rows, a, b, rho, beta, gammas=(0.0, 1.0, 3.0), q=0.25, r=0.25):
    """Use the prescribed grid, at most one expansion, retaining failed attempts."""
    attempts = []
    for limit, points in ((8.0, 401), (12.0, 601)):
        result = grid_policy_filter(rows, a, b, rho, beta, gammas, q, r, limit, points)
        attempts.append(
            {
                k: result[k]
                for k in ("status", "limit", "points", "max_outside_mass")
                if k in result
            }
        )
        if result["status"] != "GRID_INADEQUATE":
            return {**result, "grid_attempts": attempts}
    return {**result, "grid_attempts": attempts}


def simulate_policy_pair(
    a, b, rho, beta, initial, innovations, measurement, uniforms, policy
):
    """Matched h=0 counterfactual; policy h_t follows y_t and affects s_(t+1)."""
    if policy not in ("recorded", "hidden", "zero"):
        raise ValueError("unknown simulation policy")
    if not len(innovations) == len(measurement) == len(uniforms):
        raise ValueError("matched exogenous sequence lengths required")
    state, baseline, previous_h = float(initial), float(initial), 0.0
    rows, states, reference = [], [], []
    for t, (w, noise, uniform) in enumerate(
        zip(innovations, measurement, uniforms, strict=True)
    ):
        state = rho * state + beta * previous_h + w
        baseline = rho * baseline + w
        z = -1.0 if t % 2 == 0 else 1.0
        y = a + b * z + state + noise
        action = (
            0
            if policy == "zero"
            else int(y > 0)
            if policy == "recorded"
            else int(uniform < expit(3 * state))
        )
        rows.append(
            {
                "author_id": "simulation",
                "event_id": f"simulation:{t}",
                "role": "source",
                "t": t,
                "z": z,
                "y": float(y),
                "h": action,
            }
        )
        states.append(float(state))
        reference.append(float(baseline))
        previous_h = action
    delta = np.asarray(states) - reference
    return {
        "observed": rows,
        "states": states,
        "no_action_states": reference,
        "delta_state": delta.tolist(),
        "sum_squared_perturbation": float(delta @ delta),
        "sum_h": sum(row["h"] for row in rows),
    }

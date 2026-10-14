"""Source-fitted mixed effects and four-cell comparators for frozen F04."""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
from scipy.special import ndtri
from scipy.stats import t as student_t
from statsmodels.regression.mixed_linear_model import MixedLM

from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, grouped
from grain_research.response_sharing import (
    apply_function,
    compile_function,
    learn_response,
)


def generate_response_world(world, seed, counts=None, events=32, condition_count=4):
    """Generate paired budgets/condition interventions with separate role streams.

    S pairs share an intercept and have opposite slopes. H and L share latent
    parameters and uniform noise ranks; only the variance-matched noise law changes.
    """
    worlds = ("L", "T", "Q", "V", "S", "0", "H")
    if world not in worlds or condition_count not in (1, 2, 4) or events < 1:
        raise ValueError("unknown world/design")
    counts = counts or {"source": 300, "validation": 100, "application": 200}
    base_world = "L" if world == "H" else world
    world_number = worlds.index(base_world)
    conditions = {1: [-1.0], 2: [-1.0, 1.0], 4: list(CONDITIONS)}[condition_count]
    output = []
    truth = []
    for role_number, role in enumerate(("source", "validation", "application")):
        for index in range(counts[role]):
            parameter_index = index // 2 if world == "S" else index
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, 4, world_number, role_number, parameter_index, 0]
                )
            )
            a = float(rng.normal())
            b = float(rng.normal(0.5, 0.5))
            c = float(rng.normal(0, 0.4))
            tau = float(rng.uniform(-0.5, 0.5))
            sigma = float(rng.uniform(0.3, 1)) if world == "V" else 0.5
            if world == "S":
                b = -0.5 if index % 2 == 0 else 0.5
            if world == "0":
                b = 0.5
            noise_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 4, world_number, role_number, index, 1])
            )
            uniform = np.clip(
                noise_rng.random(events), np.finfo(float).eps, 1 - np.finfo(float).eps
            )
            noise = sigma * (
                student_t.ppf(uniform, 5) * np.sqrt(3 / 5)
                if world == "H"
                else ndtri(uniform)
            )
            author = (
                "u_"
                + hashlib.sha256(
                    f"{seed}/{base_world}/{role}/{index}".encode()
                ).hexdigest()[:16]
            )

            def mean(z, a=a, b=b, c=c, tau=tau):
                """Evaluate this author's conditional response, not a shared closure."""
                if world == "T":
                    return a + b * (z > tau)
                if world == "Q":
                    return a + b * z + c * z * z
                return a + b * z

            truth.append(
                {
                    "author_id": author,
                    "role": role,
                    "world": world,
                    "a": a,
                    "b": b,
                    "c": c,
                    "tau": tau,
                    "sigma2": sigma * sigma,
                    "means": [mean(z) for z in CONDITIONS],
                    "parameter_interpretation": "generating coefficients; linear pseudo-true coefficients differ in T/Q",
                }
            )
            for time in range(events):
                z = conditions[time % len(conditions)]
                value = float(mean(z) + noise[time])
                output.append(
                    {
                        "author_id": author,
                        "role": role,
                        "world": world,
                        "t": time,
                        "z": z,
                        "y": value,
                        "event_id": f"{author}/{world}/K{condition_count}/{time}",
                    }
                )
    return output, truth


def observed_arrays(rows, mode="linear"):
    """Build a design from observed conditions; reject repeated event addresses."""
    if not rows:
        raise ValueError("nonempty observations required")
    ids = [row["event_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate events do not add information")
    z = np.asarray([row["z"] for row in rows], float)
    y = np.asarray([row["y"] for row in rows], float)
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("finite observations required")
    if mode == "linear":
        x = np.column_stack((np.ones(len(z)), z))
    elif mode == "four_cell":
        positions = [list(CONDITIONS).index(value) for value in z]
        x = np.eye(4)[positions]
    else:
        raise ValueError(mode)
    return x, y


def psd_floor(matrix, floor=1e-6):
    """Apply the frozen spectral floor and retain raw covariance eigenvalues."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.maximum(values, floor)) @ vectors.T, values


def fit_source_moments(rows, mode="linear"):
    """Estimate random-coefficient covariance after subtracting sampling noise."""
    if not rows or any(row["role"] != "source" for row in rows):
        raise ValueError("source-only observations required")
    coefficients = []
    inverse_information = []
    rss = 0.0
    df = 0
    for person in grouped(rows).values():
        x, y = observed_arrays(person, mode)
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank != x.shape[1]:
            raise ValueError("source design does not identify every coefficient")
        residual = y - x @ beta
        rss += float(residual @ residual)
        df += len(y) - rank
        coefficients.append(beta)
        inverse_information.append(np.linalg.inv(x.T @ x))
    if len(coefficients) < 3 or df <= 0:
        raise ValueError(
            "at least three source authors and positive residual degrees of freedom required"
        )
    sigma2 = rss / df
    coefficient_array = np.asarray(coefficients)
    sampling = sigma2 * np.mean(inverse_information, axis=0)
    raw = np.cov(coefficient_array, rowvar=False) - sampling
    covariance, eigenvalues = psd_floor(raw)
    return {
        "mode": mode,
        "method": "source_noise_corrected_moments",
        "mu": coefficient_array.mean(axis=0).tolist(),
        "Sigma": covariance.tolist(),
        "sigma2": sigma2,
        "source_authors": len(coefficients),
        "noise_df": df,
        "sampling_covariance": sampling.tolist(),
        "raw_eigenvalues": eigenvalues.tolist(),
    }


def fit_source_lmm(rows):
    """Fit REML once, then one Powell retry; failures retain a labelled fallback."""
    if not rows or any(row["role"] != "source" for row in rows):
        raise ValueError("REML must only read source rows")
    x, y = observed_arrays(rows)
    groups = np.array([row["author_id"] for row in rows])
    model = MixedLM(y, x, groups=groups, exog_re=x)
    attempts = []
    for optimizer in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(reml=True, method=optimizer, maxiter=200, disp=False)
            covariance = np.asarray(result.cov_re, float)
            valid = bool(
                result.converged
                and np.isfinite(result.fe_params).all()
                and np.isfinite(covariance).all()
                and np.isfinite(result.scale)
                and result.scale > 0
                and np.linalg.eigvalsh(covariance).min() >= -1e-8
            )
            attempts.append(
                {
                    "optimizer": optimizer,
                    "converged": bool(result.converged),
                    "numerically_valid": valid,
                    "warnings": [str(w.message) for w in caught],
                }
            )
            if valid:
                return {
                    "mode": "linear",
                    "method": "source_REML_BLUP",
                    "mu": result.fe_params.tolist(),
                    "Sigma": covariance.tolist(),
                    "sigma2": float(result.scale),
                    "source_authors": len(set(groups)),
                    "attempts": attempts,
                    "fallback": False,
                }
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            attempts.append(
                {"optimizer": optimizer, "converged": False, "error": repr(error)}
            )
    fallback = fit_source_moments(rows)
    return {
        **fallback,
        "method": "source_moments_after_REML_failure",
        "attempts": attempts,
        "fallback": True,
    }


def score_response(prefix, fit, conditions=CONDITIONS):
    """Use a frozen source Gaussian model, retaining rank and prior contribution."""
    x, y = observed_arrays(prefix, fit["mode"])
    mu = np.asarray(fit["mu"])
    covariance = np.asarray(fit["Sigma"])
    v = x @ covariance @ x.T + fit["sigma2"] * np.eye(len(y))
    weights = np.linalg.solve(v, x @ covariance).T
    beta = mu + weights @ (y - x @ mu)
    posterior = covariance - weights @ x @ covariance
    target = (
        np.column_stack((np.ones(len(conditions)), conditions))
        if fit["mode"] == "linear"
        else np.eye(4)[[list(CONDITIONS).index(z) for z in conditions]]
    )
    return {
        "coefficients": beta,
        "means": target @ beta,
        "covariance": posterior,
        "source_term": target @ (mu - weights @ x @ mu),
        "observed_term": target @ weights @ y,
        "data_rank": int(np.linalg.matrix_rank(x)),
        "model_columns": x.shape[1],
        "data_identifies_all_coefficients": np.linalg.matrix_rank(x) == x.shape[1],
    }


def fit_source_halves(rows):
    """Reuse the historical double-half method on equal per-cell source budgets."""
    if not rows or any(row["role"] != "source" for row in rows):
        raise ValueError("source-only observations required")
    halves = []
    budgets = []
    for person in grouped(rows).values():
        values = []
        for z in CONDITIONS:
            cell = np.array([row["y"] for row in person if row["z"] == z])
            if len(cell) < 2 or len(cell) % 2:
                raise ValueError("source cells require even, nonzero half budgets")
            budgets.append(len(cell) // 2)
            values.append([float(cell[::2].mean()), float(cell[1::2].mean())])
        halves.append(values)
    if len(set(budgets)) != 1:
        raise ValueError("unequal source half budgets need a separate noise model")
    return {
        "model": learn_response(np.array(halves), np.array(CONDITIONS)),
        "half_events": budgets[0],
    }


def score_source_halves(prefix, fit):
    """Transport source half-mean noise to actual application cell-mean budgets."""
    observed_arrays(prefix)
    counts = np.array([sum(row["z"] == z for row in prefix) for z in CONDITIONS])
    indices = np.flatnonzero(counts)
    values = np.array(
        [
            np.mean([row["y"] for row in prefix if row["z"] == CONDITIONS[i]])
            for i in indices
        ]
    )
    scale = np.sqrt(
        fit["half_events"] / np.where(counts > 0, counts, fit["half_events"])
    )
    model = {
        **fit["model"],
        "noise": np.asarray(fit["model"]["noise"]) * scale[:, None] * scale[None, :],
    }
    program = compile_function(model, indices)
    result = apply_function(program, values[None, :])
    return {
        "means": result["value"][0],
        "source_noise_scale": scale,
        "observed_cells": indices,
        "program": program,
        "noise_assumption": "iid event replication; source cross-cell noise scaled by half/application counts",
    }

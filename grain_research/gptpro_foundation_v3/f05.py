"""Finite latent-state dynamics and filtering contracts for F05."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


def rho_from_eta(eta: float) -> float:
    """Map unconstrained eta to a stationary AR coefficient."""
    return float(0.99 * np.tanh(eta))


def eta_from_rho(rho: float) -> float:
    """Inverse of the bounded rho transform for |rho| < .99."""
    return float(np.arctanh(np.clip(rho / 0.99, -0.999999, 0.999999)))


def transition_covariance(rho: float, sigma_w: float, length: int) -> np.ndarray:
    """Return the stationary latent covariance for a finite sequence."""
    idx = np.arange(length)
    variance = sigma_w**2 / max(1.0 - rho**2, 1e-8)
    return variance * rho ** np.abs(idx[:, None] - idx[None, :])


def kalman_filter(y: np.ndarray, z: np.ndarray, *, a: float, b: float, rho: float, sigma_w: float, sigma_e: float) -> dict:
    """Run scalar Kalman filtering and return state means, variances and loglik."""
    q = float(sigma_w**2); r = float(sigma_e**2)
    state = 0.0
    variance = q / max(1.0 - rho**2, 1e-8)
    means, variances = [], []
    loglik = 0.0
    for observation, condition in zip(y, z):
        predicted_state = rho * state
        predicted_variance = rho**2 * variance + q
        mean_y = a + b * condition + predicted_state
        innovation = observation - mean_y
        innovation_variance = predicted_variance + r
        gain = predicted_variance / max(innovation_variance, 1e-12)
        state = predicted_state + gain * innovation
        variance = (1.0 - gain) * predicted_variance
        means.append(float(state)); variances.append(float(variance))
        loglik += -0.5 * (np.log(2 * np.pi * innovation_variance) + innovation**2 / innovation_variance)
    return {"state_mean": means, "state_variance": variances, "loglik": float(loglik)}


def dense_loglik(y: np.ndarray, z: np.ndarray, *, a: float, b: float, rho: float, sigma_w: float, sigma_e: float) -> float:
    """Compute the same Gaussian observation likelihood with dense covariance."""
    cov = transition_covariance(rho, sigma_w, len(y)) + np.eye(len(y)) * sigma_e**2
    mean = a + b * z
    residual = y - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("-inf")
    return float(-0.5 * (len(y) * np.log(2 * np.pi) + logdet + residual @ np.linalg.solve(cov, residual)))


def opaque_author(seed: int, world: str, role: str, index: int) -> str:
    """Create an opaque author key."""
    return "u_" + hashlib.sha256(f"{seed}|{world}|{role}|{index}".encode()).hexdigest()[:16]


def generate_world(world: str, seed: int, *, counts: dict[str, int] | None = None, events: int = 64) -> tuple[list[dict], list[dict]]:
    """Generate AR(1) state worlds with core and near-unit-root regimes."""
    if world not in {"core", "near_root", "shared_shock"}:
        raise ValueError(f"unknown world {world}")
    counts = counts or {"source": 40, "validation": 12, "application": 20}
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for role in ("source", "validation", "application"):
        for number in range(counts[role]):
            author = opaque_author(seed, world, role, number)
            a, b = float(rng.normal(0, 1)), float(rng.normal(.5, .5))
            rho = float(rng.uniform(.2, .8) if world != "near_root" else rng.uniform(.9, .98))
            sigma_w = sigma_e = .5
            state = float(rng.normal(0, sigma_w / np.sqrt(max(1 - rho**2, 1e-8))))
            truth.append({"author_id": author, "role": role, "world": world, "a": a, "b": b, "rho": rho, "sigma_w": sigma_w, "sigma_e": sigma_e})
            shock = 0.0
            for t in range(events):
                z = -1.0 if (t // 4) % 2 == 0 else 1.0
                if world == "shared_shock":
                    if t % 8 == 0:
                        shock = float(rng.normal(0, .6))
                    state = rho * state + shock + float(rng.normal(0, sigma_w))
                else:
                    state = rho * state + float(rng.normal(0, sigma_w))
                y = a + b * z + state + float(rng.normal(0, sigma_e))
                rows.append({"schema_version": "grain-gptpro-record-v3", "author_id": author, "role": role, "world": world, "event_id": f"{world}_{seed}_{role}_{number}_{t:04d}", "t": t, "z": z, "y": y})
    return rows, truth


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by author."""
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["author_id"]].append(row)
    return {key: sorted(value, key=lambda row: row["t"]) for key, value in result.items()}


def estimate_source_params(rows: list[dict]) -> dict:
    """Estimate a conservative shared a/b/rho profile from source rows."""
    per = []
    for author, author_rows in grouped(rows).items():
        z = np.asarray([row["z"] for row in author_rows], dtype=float)
        y = np.asarray([row["y"] for row in author_rows], dtype=float)
        x = np.column_stack([np.ones(len(z)), z])
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        rho = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if len(residual) > 2 else .5
        per.append((float(beta[0]), float(beta[1]), float(np.clip(rho, .2, .98)), float(np.std(residual))))
    arr = np.asarray(per, dtype=float)
    return {"a": float(np.mean(arr[:, 0])), "b": float(np.mean(arr[:, 1])), "rho": float(np.clip(np.mean(arr[:, 2]), .2, .98)), "sigma_w": .5, "sigma_e": float(max(.05, np.mean(arr[:, 3]) / np.sqrt(2))), "n_authors": len(per)}

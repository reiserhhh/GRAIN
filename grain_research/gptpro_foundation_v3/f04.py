"""Response-level worlds, information geometry, and BLUP probes for F04."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


CONDITIONS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
WORLD_NAMES = ("L", "T", "Q", "V", "S", "0", "H")


def design_matrix(design: str = "linear") -> np.ndarray:
    """Return the declared design matrix for four condition levels."""
    z = np.asarray(CONDITIONS, dtype=float)
    if design == "linear":
        return np.column_stack([np.ones(len(z)), z])
    if design == "quadratic":
        return np.column_stack([np.ones(len(z)), z, z**2])
    if design == "threshold":
        return np.column_stack([np.ones(len(z)), (z > 0).astype(float)])
    if design == "four_cell":
        return np.eye(len(z))
    raise ValueError(f"unknown design {design}")


def information(design: str = "linear", sigma: float = 0.5) -> dict:
    """Return rank and Fisher information for a finite condition design."""
    x = design_matrix(design)
    info = x.T @ x / float(sigma**2)
    return {"design": design, "rank": int(np.linalg.matrix_rank(x)), "columns": int(x.shape[1]), "information": info.tolist(), "sigma": float(sigma)}


def opaque_author(seed: int, world: str, role: str, index: int) -> str:
    """Create an opaque author key."""
    return "u_" + hashlib.sha256(f"{seed}|{world}|{role}|{index}".encode()).hexdigest()[:16]


def _t5(rng: np.random.Generator, size: int) -> np.ndarray:
    """Generate standardized t5 noise with unit variance."""
    values = rng.standard_t(5, size=size)
    return values * np.sqrt(3.0 / 5.0)


def generate_world(world: str, seed: int, *, counts: dict[str, int] | None = None, events: int = 32) -> tuple[list[dict], list[dict]]:
    """Generate source/validation/application response rows and truth."""
    if world not in WORLD_NAMES:
        raise ValueError(f"unknown world {world}")
    counts = counts or {"source": 300, "validation": 100, "application": 200}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    truth: list[dict] = []
    for role in ("source", "validation", "application"):
        for number in range(counts[role]):
            author = opaque_author(seed, world, role, number)
            a = float(rng.normal(0.0, 1.0))
            b = float(rng.normal(0.5, 0.5))
            c = float(rng.normal(0.0, 0.4))
            tau = float(rng.uniform(-0.5, 0.5))
            sigma_u = float(rng.uniform(0.3, 1.0)) if world == "V" else 0.5
            slope = float(rng.choice([-0.5, 0.5])) if world == "S" else b
            truth.append({"author_id": author, "role": role, "world": world, "a": a, "b": b, "c": c, "tau": tau, "sigma_u": sigma_u, "slope": slope})
            for t in range(events):
                z = CONDITIONS[t % len(CONDITIONS)]
                if world in ("L", "V", "S", "0"):
                    mean = a + (0.5 if world == "0" else slope) * z
                elif world == "T":
                    mean = a + b * float(z > tau)
                elif world == "Q":
                    mean = a + b * z + c * z**2
                else:  # H shares the linear mean but has t5 noise.
                    mean = a + b * z
                noise_scale = sigma_u if world == "V" else 0.5
                noise = float(_t5(rng, 1)[0] * noise_scale) if world == "H" else float(rng.normal(0.0, noise_scale))
                rows.append({"schema_version": "grain-gptpro-record-v3", "author_id": author, "role": role, "world": world, "event_id": f"{world}_{seed}_{role}_{number}_{t:04d}", "t": t, "z": z, "y": mean + noise})
    return rows, truth


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    """Group response rows by author."""
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["author_id"]].append(row)
    return {key: sorted(value, key=lambda row: row["t"]) for key, value in grouped_rows.items()}


def ols_fit(rows: list[dict], design: str = "linear") -> dict:
    """Fit OLS for a response world."""
    if not rows:
        return {"beta": [], "sigma2": None, "rank": 0}
    x = np.asarray([design_matrix(design)[list(CONDITIONS).index(float(row["z"]))] for row in rows], dtype=float)
    y = np.asarray([row["y"] for row in rows], dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    return {"beta": beta.tolist(), "sigma2": float(np.mean(residual**2)), "rank": int(rank), "n": len(rows), "design": design}


def fit_source_blup(source_rows: list[dict], design: str = "linear") -> dict:
    """Fit source mean/covariance and residual variance for random slopes."""
    by_author = grouped(source_rows)
    per_author = [ols_fit(rows, design) for rows in by_author.values()]
    beta = np.asarray([fit["beta"] for fit in per_author], dtype=float)
    mu = np.mean(beta, axis=0)
    sigma = np.cov(beta, rowvar=False) if len(beta) > 1 else np.eye(beta.shape[1]) * 1e-6
    sigma = np.atleast_2d(sigma) + np.eye(beta.shape[1]) * 1e-6
    sigma2 = float(np.mean([fit["sigma2"] for fit in per_author]))
    return {"design": design, "mu": mu.tolist(), "Sigma": sigma.tolist(), "sigma2": sigma2, "n_authors": len(per_author), "source_fits": per_author}


def blup_score(prefix: list[dict], fit: dict) -> dict:
    """Compute random-coefficient BLUP from a prefix."""
    if not prefix:
        return {"beta": fit["mu"], "prediction": None}
    design = fit["design"]
    x = np.asarray([design_matrix(design)[list(CONDITIONS).index(float(row["z"]))] for row in prefix], dtype=float)
    y = np.asarray([row["y"] for row in prefix], dtype=float)
    mu = np.asarray(fit["mu"], dtype=float)
    sigma = np.asarray(fit["Sigma"], dtype=float)
    sigma2 = max(float(fit["sigma2"]), 1e-8)
    post = mu + sigma @ x.T @ np.linalg.pinv(x @ sigma @ x.T + sigma2 * np.eye(len(prefix))) @ (y - x @ mu)
    return {"beta": post.tolist(), "prediction": (x @ post).tolist()}


def four_cell_fit(rows: list[dict]) -> dict:
    """Fit a saturated four-condition mean model."""
    cell = {}
    for z in CONDITIONS:
        values = [row["y"] for row in rows if float(row["z"]) == z]
        cell[str(z)] = float(np.mean(values)) if values else None
    return cell


def tail_probability(residual: np.ndarray, sigma: float) -> float:
    """Return empirical exceedance probability at 2 sigma."""
    return float(np.mean(np.abs(residual) > 2.0 * sigma)) if len(residual) else float("nan")

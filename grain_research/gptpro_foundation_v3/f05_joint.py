"""Same-episode joint state filtering with source-estimated common innovations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from grain_research.gptpro_foundation_v3.f05 import grouped


def learn_common_variance(rows, source_model):
    """Estimate average off-diagonal innovation covariance from source prefixes.

    Measurement errors are independent across authors. Parameter estimation and
    centering can still bias this moment; raw and PSD-constrained values are kept.
    """
    if any(r["role"] != "source" for r in rows):
        raise ValueError("source-only common variance")
    people = grouped(rows)
    fits = source_model["rounds"][-1]["personal_fits"]
    innovations = []
    expected_times = None
    for author, events in sorted(people.items()):
        events = events[:48]
        times = [r["t"] for r in events]
        if expected_times is None:
            expected_times = times
        if times != expected_times:
            raise ValueError("source authors must share a time grid")
        a, b, eta = fits[author]["psi"]
        rho = 0.99 * np.tanh(eta)
        residual = np.array([r["y"] - a - b * r["z"] for r in events])
        innovations.append(residual[1:] - rho * residual[:-1])
    matrix = np.asarray(innovations)
    n, t = matrix.shape
    if n < 2 or t < 2:
        raise ValueError("insufficient source replication")
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    raw = float(
        np.sum(centered.sum(axis=0) ** 2 - np.sum(centered**2, axis=0))
        / (n * (n - 1) * (t - 1))
    )
    total = float(np.exp(2 * source_model["log_sigma"][0]))
    common = float(np.clip(raw, 0, 0.99 * total))
    return {
        "raw_common_variance": raw,
        "common_variance": common,
        "total_variance": total,
        "own_variance": total - common,
        "clipped": raw != common,
        "source_authors": n,
        "innovation_times": t,
        "method": "source mean off-diagonal centered innovation covariance",
    }


def joint_filter(y, z, psi, q_total, r, common=0.0):
    """Filter a time-by-author panel using current/past observed rows only.

    Same individual parameters and marginal Q are used for joint and independent
    arms. Only off-diagonal process covariance differs. No future input enters
    earlier outputs. Covariance histories are omitted to keep full panels small.
    """
    y, z, psi = np.asarray(y), np.asarray(z), np.asarray(psi)
    if y.shape != z.shape or y.ndim != 2 or psi.shape != (y.shape[1], 3):
        raise ValueError("panel/parameter shape mismatch")
    if (
        not np.isfinite(y).all()
        or not np.isfinite(z).all()
        or not np.isfinite(psi).all()
    ):
        raise ValueError("nonfinite panel")
    if not 0 <= common <= q_total or q_total <= 0 or r <= 0:
        raise ValueError("invalid noise covariance")
    rho = 0.99 * np.tanh(psi[:, 2])
    size = y.shape[1]
    q = np.full((size, size), common)
    np.fill_diagonal(q, q_total)
    transition = rho[:, None] * rho[None, :]
    covariance = q / (1 - transition)
    mean = np.zeros(size)
    means, variances = [], []
    for t in range(len(y)):
        if t:
            mean = rho * mean
            covariance = transition * covariance + q
        innovation = y[t] - psi[:, 0] - psi[:, 1] * z[t] - mean
        factor = cho_factor(
            covariance + r * np.eye(size), lower=True, check_finite=False
        )
        gain = cho_solve(factor, covariance, check_finite=False).T
        mean = mean + gain @ innovation
        covariance = covariance - gain @ covariance
        covariance = (covariance + covariance.T) / 2
        means.append(mean.copy())
        variances.append(np.diag(covariance).copy())
    return {"state_mean": np.asarray(means), "state_variance": np.asarray(variances)}

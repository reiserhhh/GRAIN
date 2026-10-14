#!/usr/bin/env python3
"""Independent dense joint-Gaussian reference for a requested target clock."""

import numpy as np


def dense_moments(parameters, rows, log_sigma, cutoff, target):
    """Condition full covariance matrices without filtering or propagation code."""
    p = np.atleast_2d(np.asarray(parameters, float))
    if target not in ("T2", "T3"):
        raise ValueError("T2/T3 target required")
    rho = 0.99 * np.tanh(p[:, 2])
    q, r = np.exp(2 * np.asarray(log_sigma, float))
    stationary = q / (1 - rho * rho)
    future = np.array([cutoff]) if target == "T2" else cutoff + np.arange(1, 5)
    z = np.array([0]) if target == "T2" else np.array([1, -1, 1, -1])
    prior = stationary[:, None, None] * rho[:, None, None] ** np.abs(
        future[:, None] - future[None, :]
    )
    mean = np.zeros((len(p), len(future)))
    if target == "T3":
        prior += r * np.eye(4)
        mean = p[:, 0, None] + p[:, 1, None] * z
    if rows:
        times = np.array([row["t"] for row in rows])
        if np.any(np.diff(times) <= 0) or cutoff < times[-1]:
            raise ValueError("ordered history must precede target cutoff")
        past = stationary[:, None, None] * rho[:, None, None] ** np.abs(
            times[:, None] - times[None, :]
        )
        past += r * np.eye(len(times))
        cross = stationary[:, None, None] * rho[:, None, None] ** np.abs(
            future[:, None] - times[None, :]
        )
        residual = (
            np.array([row["y"] for row in rows])[None, :]
            - p[:, 0, None]
            - p[:, 1, None] * np.array([row["z"] for row in rows])
        )
        rhs = np.concatenate([residual[:, :, None], cross.transpose(0, 2, 1)], axis=2)
        solved = np.linalg.solve(past, rhs)
        mean += np.einsum("dkn,dn->dk", cross, solved[:, :, 0])
        prior -= cross @ solved[:, :, 1:]
    return mean.mean(axis=1), prior.sum(axis=(1, 2)) / len(future) ** 2

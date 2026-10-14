"""Exact observational equivalences and local rank diagnostics for F05."""

from __future__ import annotations

import numpy as np


def stationary_covariance(a, q):
    """Solve discrete Lyapunov by vectorization, without conjugate transpose."""
    dimension = len(a)
    return np.linalg.solve(
        np.eye(dimension**2) - np.kron(a, a), q.reshape(-1, order="F")
    ).reshape((dimension, dimension), order="F")


def autocovariances(a, q, c, r, lags=16):
    """Stationary scalar-output covariance, including observation noise at zero."""
    p = stationary_covariance(a, q)
    return np.asarray(
        [
            (c @ np.linalg.matrix_power(a, k) @ p @ c.T).item() + (r if k == 0 else 0)
            for k in range(lags + 1)
        ]
    )


def observation_covariance(a, q, c, r, times):
    """Full stationary Gaussian law for arbitrary integer observation times."""
    difference = np.abs(np.asarray(times)[:, None] - np.asarray(times)[None, :])
    gamma = autocovariances(a, q, c, r, int(difference.max()))
    # Noise is per observation, not per time lag; duplicate times not allowed.
    if len(np.unique(times)) != len(times):
        raise ValueError("distinct observation times required")
    return gamma[difference]


def similarity_transform(a, q, c, p, s):
    """Transport all state-space objects under x'=Sx."""
    inverse = np.linalg.inv(s)
    return s @ a @ inverse, s @ q @ s.T, c @ inverse, s @ p @ s.T


def conditional_future(a, q, c, r, history, horizon=4):
    """Condition future observations on past values with the full Gaussian law."""
    n = len(history)
    covariance = observation_covariance(a, q, c, r, np.arange(n + horizon))
    past, cross = covariance[:n, :n], covariance[n:, :n]
    mean = cross @ np.linalg.solve(past, history)
    posterior = covariance[n:, n:] - cross @ np.linalg.solve(past, cross.T)
    return mean, posterior


def covariance_jacobian(a, q, c, r, free="all", lags=16):
    """Complex-step derivatives of output autocovariances with explicit freedoms.

    Rank is local at the supplied model and is not a global identification proof.
    All: A(4), symmetric Q(3), C(2), R(1). Fixed_CQ: A(4), R(1).
    Fixed_CQR: A(4). No noise covariance or output loading is silently refitted.
    """
    if free not in {"all", "fixed_CQ", "fixed_CQR"}:
        raise ValueError("unknown parameter restrictions")
    base = list(a.ravel())
    if free == "all":
        base += [q[0, 0], q[0, 1], q[1, 1], *c.ravel(), r]
    elif free == "fixed_CQ":
        base += [r]

    def unpack(theta):
        aa = theta[:4].reshape(2, 2)
        if free == "all":
            qq = np.array([[theta[4], theta[5]], [theta[5], theta[6]]])
            cc = theta[7:9].reshape(1, 2)
            rr = theta[9]
        else:
            qq, cc = q, c
            rr = theta[4] if free == "fixed_CQ" else r
        return autocovariances(aa, qq, cc, rr, lags)

    columns = []
    for index in range(len(base)):
        theta = np.asarray(base, dtype=complex)
        theta[index] += 1e-25j
        columns.append(np.imag(unpack(theta)) / 1e-25)
    jacobian = np.column_stack(columns)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = singular[0] * 1e-9
    return {
        "jacobian": jacobian,
        "singular_values": singular,
        "rank": int(np.sum(singular > tolerance)),
        "parameters": len(base),
        "tolerance": tolerance,
    }

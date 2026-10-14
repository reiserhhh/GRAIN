"""Finite information-order certificates for the frozen F15 protocol."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def _channel(value: object) -> np.ndarray:
    """Validate a target-by-observation row-stochastic probability matrix."""
    channel = np.asarray(value, dtype=float)
    if channel.ndim != 2 or min(channel.shape) < 1:
        raise ValueError("a nonempty target-by-observation matrix is required")
    if not np.isfinite(channel).all() or (channel < 0).any():
        raise ValueError("channel probabilities must be finite and nonnegative")
    if not np.allclose(channel.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("channel rows must sum to one")
    return channel


def garbling_certificate(a: object, b: object) -> dict:
    """Solve B=A K with a row-stochastic K; certify infeasibility by Farkas.

    Rows of A/B refer to the same declared target states. For an infeasible
    system E k=d, k>=0, find y with E.T@y>=0 and d@y=-1. Such a vector is a
    checkable contradiction, unlike merely reporting a solver status message.
    """
    a, b = _channel(a), _channel(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("both channels must use the same target states")
    n, m = a.shape[1], b.shape[1]
    # C-order flattening gives vec_C(A K)=(A kron I_m) vec_C(K).
    equations = np.vstack((np.kron(a, np.eye(m)), np.kron(np.eye(n), np.ones((1, m)))))
    rhs = np.concatenate((b.ravel(), np.ones(n)))
    result = linprog(
        np.zeros(n * m), A_eq=equations, b_eq=rhs, bounds=(0, None), method="highs"
    )
    if result.success:
        kernel = result.x.reshape(n, m)
        residual = float(np.max(np.abs(equations @ result.x - rhs)))
        valid = residual <= 1e-10 and float(kernel.min()) >= -1e-10
        return {
            "status": "FEASIBLE" if valid else "NUMERICAL_FAILURE",
            "kernel": kernel.tolist(),
            "max_equation_residual": residual,
            "min_kernel": float(kernel.min()),
            "solver_status": int(result.status),
        }
    if result.status != 2:
        return {
            "status": "NUMERICAL_FAILURE",
            "solver_status": int(result.status),
            "message": result.message,
        }
    dual = linprog(
        np.zeros(equations.shape[0]),
        A_ub=-equations.T,
        b_ub=np.zeros(equations.shape[1]),
        A_eq=rhs[None, :],
        b_eq=[-1.0],
        bounds=(None, None),
        method="highs",
    )
    if not dual.success:
        return {
            "status": "UNVERIFIED_INFEASIBLE",
            "solver_status": int(result.status),
            "dual_message": dual.message,
        }
    minimum = float(np.min(equations.T @ dual.x))
    dot = float(rhs @ dual.x)
    valid = minimum >= -1e-10 and dot < -1e-10
    return {
        "status": "INFEASIBLE_CERTIFIED" if valid else "NUMERICAL_FAILURE",
        "farkas_y": dual.x.tolist(),
        "min_dual_constraint": minimum,
        "rhs_dot_y": dot,
        "solver_status": int(result.status),
    }


def mutual_information(channel: object, prior: object) -> float:
    """Compute exact finite I(target; observation) in nats, with 0 log 0=0."""
    channel = _channel(channel)
    prior = np.asarray(prior, dtype=float)
    if (
        prior.shape != (channel.shape[0],)
        or not np.isfinite(prior).all()
        or (prior < 0).any()
    ):
        raise ValueError("invalid target prior")
    if not np.isclose(prior.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("target prior must sum to one")
    joint = prior[:, None] * channel
    product = prior[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / product[mask])))

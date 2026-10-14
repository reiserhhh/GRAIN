"""Explicit edit-chart action geometry, separate from native text interventions."""

import numpy as np
from scipy.linalg import eigh


def quadratic_response(e, h0, a, u, b):
    """Evaluate queries at h=h0+A e; B's symmetric part defines each quadratic."""
    h = np.asarray(h0) + np.asarray(e) @ np.asarray(a).T
    return h @ np.asarray(u).T + 0.5 * np.einsum("...i,qij,...j->...q", h, b, h)


def edit_jacobian(e, h0, a, u, b):
    """Return query-by-edit derivatives using the full hidden-to-edit chain rule."""
    h = np.asarray(h0) + np.asarray(a) @ np.asarray(e)
    sym = (np.asarray(b) + np.asarray(b).swapaxes(-1, -2)) / 2
    return (np.asarray(u) + np.einsum("qij,j->qi", sym, h)) @ np.asarray(a)


def common_metric(jacobians, weights=None):
    """Build PSD sum_q w_q J_q'J_q; never collapse a single gradient to a scalar."""
    j = np.asarray(jacobians, dtype=float)
    if j.ndim == 1:
        j = j[None, :]
    if j.ndim != 2 or not np.isfinite(j).all() or 0 in j.shape:
        raise ValueError("finite nonempty query-by-edit Jacobian required")
    w = (
        np.ones(len(j)) / len(j)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if (
        w.shape != (len(j),)
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or w.sum() <= 0
    ):
        raise ValueError("nonnegative finite query weights with positive sum required")
    return j.T @ (w[:, None] * j)


def action_subspace(jacobians, source_covariance, dimension=2):
    """Return source-frozen generalized eigenspace and SVD numerical nullspace."""
    j = np.atleast_2d(np.asarray(jacobians, dtype=float))
    g = common_metric(j)
    c = np.asarray(source_covariance, dtype=float)
    if c.shape != g.shape or not np.allclose(c, c.T) or not np.isfinite(c).all():
        raise ValueError("finite symmetric source covariance required")
    if np.linalg.eigvalsh(c).min() < -1e-10 or not 1 <= dimension <= g.shape[0]:
        raise ValueError("PSD covariance and valid dimension required")
    metric = c + 1e-6 * np.eye(len(c))
    values, vectors = eigh(g, metric)
    _, singular, vt = np.linalg.svd(j, full_matrices=True)
    tolerance = max(j.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    return {
        "metric": g,
        "denominator": metric,
        "values": values[::-1],
        "basis": vectors[:, ::-1][:, :dimension],
        "rank": rank,
        "tolerance": tolerance,
        "nullspace": vt[rank:].T,
    }


def project_edits(edits, basis, metric=None):
    """Project onto a frozen basis in Euclidean or specified positive metric."""
    v = np.asarray(basis)
    m = np.eye(v.shape[0]) if metric is None else np.asarray(metric)
    coefficients = np.linalg.solve(v.T @ m @ v, v.T @ m @ np.asarray(edits).T)
    return (v @ coefficients).T

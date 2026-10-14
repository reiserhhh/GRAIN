#!/usr/bin/env python3
"""Globally coherent shared-reference comparisons of frozen SOURCE laws."""

import numpy as np


def validate_mass(p, q):
    """Require two positive categorical laws without repairing their masses."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if (
        p.ndim != 1
        or q.shape != p.shape
        or not len(p)
        or not np.isfinite([p, q]).all()
        or np.any(p <= 0)
        or np.any(q <= 0)
        or not np.allclose([p.sum(), q.sum()], 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Positive normalized laws on one SOURCE alphabet required")
    return p, q


def coupling(p, q):
    """Integrate the common exponential race, equivalent to shared Gumbel-max.

    For r_i=p_i/q_i and S(t)=sum_k max(p_k,t*q_k), off-diagonal
    J_ij=p_i*q_j*integral_(r_j)^(r_i) S(t)^(-2) dt when r_i>r_j.
    Diagonal J_ii=p_i/S(r_i). One noise vector supplies all authors;
    these are its pair marginals, not separately optimized transports.
    """
    p, q = validate_mass(p, q)
    ratio = p / q
    if not np.isfinite(ratio).all():
        raise ValueError("SOURCE probability ratios exceed finite numeric support")
    order = np.argsort(ratio, kind="stable")
    r, ps, qs = ratio[order], p[order], q[order]
    suffix = np.cumsum(ps[::-1])[::-1]
    prefix = np.r_[0, np.cumsum(qs)[:-1]]
    s = suffix + r * prefix
    integral = np.r_[0, np.cumsum(np.diff(r) / (s[:-1] * s[1:]))]
    f, at_ratio = np.empty_like(p), np.empty_like(p)
    f[order], at_ratio[order] = integral, s
    joint = p[:, None] * q[None] * np.maximum(f[:, None] - f[None], 0)
    np.fill_diagonal(joint, p / at_ratio)
    return joint


def kernels(anchors):
    """Keep each SOURCE author's full anchor row in all target comparisons."""
    y = np.asarray(anchors, float)
    if y.ndim != 2 or y.shape[1] != 5 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Five finite joint SOURCE anchors required")
    diff = y[:, None] - y[None]
    return (diff > 0).astype(float) + 0.5 * (diff == 0)


def accounts(p, q, anchors, comparison_kernels=None):
    """Return comparison and cross-author dependence, keeping marginal means."""
    p, q = validate_mass(p, q)
    y = np.asarray(anchors, float)
    k = kernels(y) if comparison_kernels is None else comparison_kernels
    if y.shape != (len(p), 5) or k.shape != (len(p), len(p), 5):
        raise ValueError("SOURCE anchors and comparison alphabet disagree")
    j = coupling(p, q)
    probability = np.einsum("ij,ijq->q", j, k)
    same = float(np.trace(j))
    off = j.copy()
    np.fill_diagonal(off, 0)
    changed = float(off.sum())
    conditional = (
        np.einsum("ij,ijq->q", off, k) / changed if changed > 0 else np.full(5, np.nan)
    )
    mu, mv = p @ y, q @ y
    cross = y.T @ j @ y - np.outer(mu, mv)
    mean_replay = j.sum(1) @ y - j.sum(0) @ y
    return {
        "probability": probability,
        "same_source_probability": same,
        "changed_source_probability": changed,
        "conditional_probability_given_changed_source": conditional,
        "cross_covariance": cross,
        "row_mass_error": float(abs(j.sum(1) - p).max()),
        "column_mass_error": float(abs(j.sum(0) - q).max()),
        "mean_difference_error": float(abs(mean_replay - (mu - mv)).max()),
    }


def sample_shared(mass, exponentials):
    """Map one exogenous race to all authors, without lexical or label input."""
    w, e = np.asarray(mass, float), np.asarray(exponentials, float)
    if (
        w.ndim != 2
        or e.ndim != 2
        or e.shape[1] != w.shape[1]
        or not np.isfinite(w).all()
        or not np.isfinite(e).all()
        or np.any(w <= 0)
        or np.any(e <= 0)
        or not np.allclose(w.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Positive laws and common exponential observations required")
    return np.argmin(e[:, None, :] / w[None], axis=-1)

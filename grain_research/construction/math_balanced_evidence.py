#!/usr/bin/env python3
"""Construct positive text-mixture contrasts before observing reader responses."""

import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp


def slot_indicators(removals, events=12, kept=8):
    """Encode each real event at each rendered slot; never invent a vector corner."""
    s = np.asarray(removals)
    if (
        s.ndim != 2
        or s.shape[1] != events - kept
        or s.dtype.kind not in "iu"
        or np.any(s < 0)
        or np.any(s >= events)
        or np.any(np.diff(s, axis=1) <= 0)
    ):
        raise ValueError("Ordered, distinct in-range removals required")
    z = np.zeros((len(s), events, kept))
    for k, row in enumerate(s):
        retained = [i for i in range(events) if i not in row]
        z[k, retained, np.arange(kept)] = 1
    return z


def construct(accounts, removals, conditions, condition_count, *, target=3):
    """Maximize a SOURCE target contrast with positive arms and fixed rival moments."""
    a, s, c = np.asarray(accounts, float), np.asarray(removals), np.asarray(conditions)
    slots = slot_indicators(s)
    if (
        a.shape != (len(s), 3, 5)
        or not np.isfinite(a).all()
        or np.max(abs(a[:, 0] - a[:, 1] - a[:, 2])) > 1e-8
        or c.shape != (12,)
        or not np.isin(c, [0, 1]).all()
        or target not in range(5)
    ):
        raise ValueError("Complete SOURCE F/B/D and twelve conditions required")
    ix = np.flatnonzero(c[s].sum(1) == condition_count)
    if len(ix) < 2:
        raise ValueError("Insufficient fixed-condition support; no relaxation")
    rivals = [q for q in range(5) if q != target]
    z = np.column_stack(
        (slots[ix].reshape(len(ix), -1), a[ix][:, :, rivals].reshape(len(ix), -1))
    )
    # Centering/rescaling constraints is algebraic preconditioning, not text denoising.
    center = z.mean(0)
    scale = np.ptp(z, axis=0)
    scale[scale == 0] = 1
    zs = (z - center) / scale
    n = len(ix)
    equality = np.vstack(
        (
            np.r_[np.ones(n), np.zeros(n)],
            np.r_[np.zeros(n), np.ones(n)],
            np.column_stack((zs.T, -zs.T)),
        )
    )
    rhs = np.r_[1.0, 1.0, np.zeros(z.shape[1])]
    g = a[ix, 2, target]
    objective = np.r_[-g, g]
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=rhs,
        bounds=(0, None),
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        raise RuntimeError(f"Finite SOURCE design failed: {result.message}; no retry")
    p = result.x.reshape(2, n)
    if np.min(p) < 0 or np.max(abs(p.sum(1) - 1)) > 1e-8:
        raise ValueError("Invalid probability arms; no clipping/renormalization")
    contrast = p[0] - p[1]
    balance = contrast @ z
    dual = np.asarray(result.eqlin.marginals)
    slack = objective - equality.T @ dual
    error = max(
        float(np.max(abs(balance))), float(np.max(abs(equality @ result.x - rhs)))
    )
    gap = float(objective @ result.x - rhs @ dual)
    if error > 1e-8 or np.min(slack) < -1e-8 or abs(gap) > 1e-8:
        raise ValueError("Invalid primal/dual certificate; no tolerance rescue")
    full = np.zeros((2, len(s)))
    full[:, ix] = p
    return {
        "weights": full,
        "candidate_indices": ix,
        "balance_features": z,
        "balance_error": error,
        "source_actions": np.einsum("k,kfq->fq", contrast, a[ix]),
        "support": np.flatnonzero(np.any(full > 0, axis=0)),
        "dual": dual,
        "dual_min_slack": float(np.min(slack)),
        "duality_gap": gap,
        "condition_count": int(condition_count),
    }


def mixture_action(weights, logits):
    """Average complete-input log odds, not logits, states or pooled probabilities."""
    p, x = np.asarray(weights, float), np.asarray(logits, float)
    if (
        p.ndim != 2
        or p.shape[0] != 2
        or x.ndim != 3
        or x.shape[0] != p.shape[1]
        or x.shape[2] != 10
        or not np.isfinite(x).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.max(abs(p.sum(1) - 1)) > 1e-8
    ):
        raise ValueError(
            "Positive arms and all selected native digit observations required"
        )
    y = logsumexp(x[..., 5:], -1) - logsumexp(x[..., :5], -1)
    arm_means = p @ y
    return arm_means, arm_means[0] - arm_means[1], y

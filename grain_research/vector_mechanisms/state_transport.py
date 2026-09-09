#!/usr/bin/env python3
"""Construction-only reduced state transport from early query slots to P/Q.

The dual kernel form avoids a hidden-dimension squared matrix. This is an
empirical reduced-state emulator, not a theorem about the transformer's circuit.
"""

from __future__ import annotations

import numpy as np


def pack_state(p, q):
    """Flatten ordered question rows and the distinct final slot without pooling."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    if p.ndim != 2 or q.shape != (p.shape[1],) or not len(p):
        raise ValueError("Ordered P rows and a matching final Q vector required")
    value = np.r_[p.ravel(), q]
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite state")
    return value


def unpack_state(vector, shape):
    """Restore exact within-query coordinates; no cross-question row matching."""
    v = np.asarray(vector, np.float64)
    rows, width = shape
    if rows < 1 or width < 1 or v.shape != ((rows + 1) * width,):
        raise ValueError("State shape mismatch")
    return v[:-width].reshape(rows, width), v[-width:]


def raw_kernel(x, y, degree):
    """Use a linear or fixed linear-plus-quadratic positive-semidefinite kernel."""
    if degree not in (1, 2):
        raise ValueError("Only the two declared kernels are permitted")
    value = x @ y.T
    return value if degree == 1 else value + 0.5 * value**2


def fit_transport(x, y, alpha=1.0):
    """Fit centered residual operators on construction snapshots, never labels.

    x and y have the same query-slot coordinates at different network depths.
    Scale and kernel centering are learned from construction snapshots only.
    """
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or x.shape != y.shape or len(x) < 3 or alpha <= 0:
        raise ValueError("Paired construction snapshots and positive alpha required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite construction snapshot")
    mean_x = x.mean(0)
    scale = float(np.sqrt(np.mean(np.sum((x - mean_x) ** 2, axis=1))))
    if scale <= 1e-12:
        raise ValueError("No early-state variation in construction")
    xn = (x - mean_x) / scale
    residual = y - x
    mean_r = residual.mean(0)
    rc = residual - mean_r
    model = {
        "mean_x": mean_x,
        "scale": np.asarray(scale),
        "train_x": xn,
        "mean_r": mean_r,
        "mean_y": y.mean(0),
        "alpha": np.asarray(alpha),
        "target_variation": np.mean((y - y.mean(0)) ** 2, axis=0),
    }
    for degree in (1, 2):
        k = raw_kernel(xn, xn, degree)
        col, grand = k.mean(0), float(k.mean())
        kc = k - col[None, :] - col[:, None] + grand
        model[f"col{degree}"] = col
        model[f"grand{degree}"] = np.asarray(grand)
        model[f"coef{degree}"] = np.linalg.solve(kc + alpha * np.eye(len(x)), rc)
        model[f"effective_df{degree}"] = np.asarray(
            np.trace(np.linalg.solve(kc + alpha * np.eye(len(x)), kc))
        )
    return model


def reconstruct(model, x):
    """Generate full P/Q-shaped states without observing any later-layer state."""
    x = np.asarray(x, np.float64)
    if x.shape != model["mean_x"].shape or not np.isfinite(x).all():
        raise ValueError("Early-state coordinate mismatch")
    z = (x - model["mean_x"]) / float(model["scale"])
    base = x + model["mean_r"]
    result = {"upper_mean": model["mean_y"].copy(), "residual_mean": base}
    for degree in (1, 2):
        k = raw_kernel(z[None, :], model["train_x"], degree)[0]
        kc = k - k.mean() - model[f"col{degree}"] + model[f"grand{degree}"]
        result[f"transport{degree}"] = base + kc @ model[f"coef{degree}"]
    return result


def state_errors(estimated, actual, model, shape):
    """Report state error against construction variation, not huge common offsets."""
    p, q = unpack_state(estimated, shape)
    tp, tq = unpack_state(actual, shape)
    vp, vq = unpack_state(model["target_variation"], shape)
    error = lambda a, b, v: {
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "construction_variation_nrmse": float(
            np.sqrt(np.mean((a - b) ** 2) / max(float(v.mean()), 1e-20))
        ),
    }
    return {
        f"{name}_{k}": value
        for name, a, b, v in (("P", p, tp, vp), ("Q", q, tq, vq))
        for k, value in error(a, b, v).items()
    }

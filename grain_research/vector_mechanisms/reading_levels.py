#!/usr/bin/env python3
"""Separate reading levels and affine error components in fixed feature geometry."""

from __future__ import annotations

import copy

import numpy as np
from sklearn.linear_model import Ridge

from .configuration_extraction import InputProgram
from .natural_transport import correlation


def split_loss(actual, predicted):
    """Exact balanced-panel pooled = between + within squared loss."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.ndim != 3 or p.shape != y.shape or min(y.shape) < 1:
        raise ValueError("Balanced author/view/target arrays required")
    if not np.isfinite([y, p]).all():
        raise ValueError("Finite arrays required")
    error = y - p
    avg = error.mean(1, keepdims=True)
    total = np.mean(error**2, axis=(0, 1))
    between = np.mean(avg**2, axis=(0, 1))
    within = np.mean((error - avg) ** 2, axis=(0, 1))
    return {
        "pooled": total.tolist(),
        "between": between.tolist(),
        "within": within.tolist(),
        "max_error": float(np.max(np.abs(total - between - within))),
    }


def fit_readers(bags, target, views=3, penalty=0.01):
    """Fit three loss objects without changing their common source feature scale."""
    y = np.asarray(target, float)
    if len(bags) % views or y.shape != (len(bags), 2) or views < 2:
        raise ValueError("Complete balanced source views required")
    if not np.isfinite(y).all() or penalty <= 0:
        raise ValueError("Finite targets and positive fixed penalty required")
    # Poles are unused by the mean-only program; no new query embeddings needed.
    pooled = InputProgram(
        "e5_mean", np.zeros((2, bags[0].shape[1])), penalty=penalty
    ).fit(bags, None, y)
    x = pooled.transform(bags, None).reshape(-1, views, bags[0].shape[1])
    yp = y.reshape(-1, views, 2)
    between = copy.deepcopy(pooled)
    between.head = Ridge(alpha=penalty * len(x)).fit(x.mean(1), yp.mean(1))
    xw = (x - x.mean(1, keepdims=True)).reshape(len(y), -1)
    yw = (yp - yp.mean(1, keepdims=True)).reshape(len(y), 2)
    within = Ridge(alpha=penalty * len(y), fit_intercept=False).fit(xw, yw)
    account = split_loss(yp, pooled.predict(bags, None).reshape(yp.shape))
    if account["max_error"] > 1e-10:
        raise ValueError("Pooled loss identity failed")
    return {
        "pooled": pooled,
        "between": between,
        "within_coef": within.coef_,
        "views": views,
        "loss_account": account,
    }


def within_predictions(readers, bags):
    """Read centered views only; their level intercept is not identified."""
    x = readers["pooled"].transform(bags, None)
    views = readers["views"]
    if len(x) % views:
        raise ValueError("Complete author view sets required")
    grouped = x.reshape(-1, views, x.shape[1])
    return (
        (grouped - grouped.mean(1, keepdims=True)) @ readers["within_coef"].T
    ).reshape(-1, 2)


def within_change(readers, full, edited):
    """Predict a material change with an identified slope, without a level score."""
    model = readers["pooled"]
    delta = model.transform(edited, None) - model.transform(full, None)
    return delta @ readers["within_coef"].T


def affine_error_account(target, prediction, criterion):
    """Descriptive alignment separates scale error from a non-affine remainder.

    Alignment uses observed target values. It is not a deployable calibration
    or a new held-out performance claim. Criterion never determines the slope.
    """
    t, p, y = (np.asarray(a, float) for a in (target, prediction, criterion))
    if t.ndim != 1 or t.shape != p.shape or t.shape != y.shape or len(t) < 4:
        raise ValueError("Matched author vectors required")
    if not np.isfinite([t, p, y]).all() or min(t.std(), p.std(), y.std()) < 1e-12:
        raise ValueError("Finite nonconstant author vectors required")
    tc, pc, yc = t - t.mean(), p - p.mean(), y - y.mean()
    slope = float(pc @ tc / (pc @ pc))
    non_affine = tc - slope * pc
    amplitude = (slope - 1) * pc
    denom = t.std() * y.std()
    cov = lambda a: float(np.mean(a * yc) / denom)
    gap, scale, remainder = cov(tc - pc), cov(amplitude), cov(non_affine)
    return {
        "n": len(t),
        "teacher_fidelity_r": correlation(t, p),
        "alignment_slope": slope,
        "total_error_criterion_contribution": gap,
        "amplitude_contribution": scale,
        "non_affine_contribution": remainder,
        "non_affine_criterion_r": correlation(non_affine, y),
        "closure_error": abs(gap - scale - remainder),
        "orthogonality_error": float(abs(np.mean(non_affine * pc))),
    }

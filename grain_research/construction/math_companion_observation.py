#!/usr/bin/env python3
"""Same-evidence companion interactions, with representation/readout accounts."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def mixed(values):
    """Return the own-minus-stranger S-minus-C action on the last two axes."""
    z = np.asarray(values, float)
    if z.shape[-2:] != (2, 2) or not np.isfinite(z).all():
        raise ValueError("Finite [..., companion(own,stranger), arm(S,C)] required")
    return z[..., 0, 0] - z[..., 0, 1] - z[..., 1, 0] + z[..., 1, 1]


def read_logits(logits):
    """Retain both conditional expected score and high/low log odds."""
    z = np.asarray(logits, float)
    if z.shape[-1] != 10 or not np.isfinite(z).all():
        raise ValueError("Ten finite candidate digit logits required")
    logq = z - logsumexp(z, axis=-1, keepdims=True)
    return {
        "score": np.exp(logq) @ (np.arange(10) / 9),
        "log_odds": logsumexp(z[..., 5:], axis=-1) - logsumexp(z[..., :5], axis=-1),
        "log_candidate_partition": logsumexp(z, axis=-1),
    }


def head_accounts(states, weights, bias):
    """Separate a four-corner head action without rerunning a decoder."""
    h, w, b = np.asarray(states, float), np.asarray(weights, float), np.asarray(bias)
    if (
        h.ndim < 3
        or h.shape[-3:-1] != (2, 2)
        or w.shape != (10, h.shape[-1])
        or b.shape != (10,)
        or not all(np.isfinite(z).all() for z in (h, w, b))
    ):
        raise ValueError("Finite companion/arm/hidden corners and frozen digit head")
    h11, h01, h10, h00 = (
        h[..., 0, 0, :],
        h[..., 0, 1, :],
        h[..., 1, 0, :],
        h[..., 1, 1, :],
    )
    additive = h10 + h01 - h00
    z = h @ w.T + b
    za = additive @ w.T + b
    f = read_logits(z)["log_odds"]
    fa = read_logits(za)["log_odds"]
    total = mixed(f)
    reader = fa - f[..., 0, 1] - f[..., 1, 0] + f[..., 1, 1]
    representation = f[..., 0, 0] - fa
    delta_z = z[..., 0, 0, :] - z[..., 0, 1, :] - z[..., 1, 0, :] + z[..., 1, 1, :]
    return {
        "logits": z,
        "total": total,
        "reader": reader,
        "representation": representation,
        "mixed_hidden": h11 - h01 - h10 + h00,
        "mixed_centered_logits": delta_z - delta_z.mean(axis=-1, keepdims=True),
        "closure": total - reader - representation,
    }


def cosine(left, right):
    """Descriptive centered contact; zero variation is missing, not success."""
    x, y = np.asarray(left, float).ravel(), np.asarray(right, float).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Matched finite contact observations required")
    x, y = x - x.mean(), y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else None


def rms(value):
    """Keep the signed observations and summarize magnitude separately."""
    return float(np.sqrt(np.mean(np.asarray(value, float) ** 2)))


def contacts(source, native):
    """All source/query contacts; authors and their two views stay paired."""
    x, y = np.asarray(source, float), np.asarray(native, float)
    if x.shape[:2] != y.shape[:2] or x.ndim != 3 or y.ndim != 3:
        raise ValueError("Author/view/target tables required")
    return [
        [cosine(x[..., t], y[..., q]) for q in range(y.shape[-1])]
        for t in range(x.shape[-1])
    ]

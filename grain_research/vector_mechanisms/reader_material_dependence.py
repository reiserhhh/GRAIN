#!/usr/bin/env python3
"""Fixed-last interventions and finite-panel reader observability accounts."""

import numpy as np

from .context_reading_interaction import replace_last
from .source_reader_motion import secant


def clamp(data, last):
    """Replace the complete last vector, preserving every non-last input row."""
    return dict(data, state=replace_last(data["state"], last, data["positions"]))


def perturb(data, direction, step):
    """Perturb only the declared BODY; retained rows and last remain bitwise fixed."""
    x, d, pos = data["state"], np.asarray(direction), data["positions"]
    allowed = np.zeros(len(x), dtype=bool)
    allowed[pos] = True
    if (
        d.shape != x.shape
        or not np.isfinite(d).all()
        or not np.isfinite(step)
        or np.any(d[~allowed])
        or allowed[-1]
    ):
        raise ValueError("Finite BODY-only direction and step required")
    y = (x.astype(float) + step * d).astype(x.dtype)
    np.testing.assert_array_equal(y[~allowed], x[~allowed])
    return dict(data, state=y)


def dispersion(rows):
    """Exact empirical best-constant SSE and pairwise reader geometry."""
    x = np.asarray(rows, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite aligned rows required")
    mean = x.mean(axis=0)
    centered, energy = x - mean, float(np.sum(x**2))
    floor = float(np.sum(centered**2))
    norms = np.linalg.norm(x, axis=1)
    cos = []
    for i in range(len(x)):
        for j in range(i):
            if norms[i] and norms[j]:
                cos.append(float(x[i] @ x[j] / (norms[i] * norms[j])))
    return {
        "authors": len(x),
        "energy": energy,
        "constant_floor_sse": floor,
        "mean_energy": float(len(x) * (mean @ mean)),
        "floor_over_energy": floor / energy if energy else 0.0,
        "centered_l2_rms": float(np.sqrt(floor / len(x))),
        "pair_cosine_mean": float(np.mean(cos)) if cos else 0.0,
        "pair_cosine_count": len(cos),
    }


def ratio(a, b):
    """Energy ratio with an explicit zero-reference convention for diagnostics."""
    aa, bb = (
        float(np.sum(np.asarray(a, float) ** 2)),
        float(np.sum(np.asarray(b, float) ** 2)),
    )
    return aa / bb if bb else 0.0


def energy_outside(rows, basis):
    """Fraction outside a fixed orthonormal row basis, not a recipient-fitted axis."""
    x, u = np.asarray(rows, float), np.asarray(basis, float)
    if u.ndim != 2 or x.shape[-1] != u.shape[1]:
        raise ValueError("Compatible row basis required")
    np.testing.assert_allclose(u @ u.T, np.eye(len(u)), atol=1e-10)
    return ratio(x - (x @ u.T) @ u, x)


def summarize_scalars(values):
    """Retain sign and amplitude over authors instead of inflating factor n."""
    x = np.asarray(values, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Finite author-indexed scalars required")
    return {
        "authors": len(x),
        "mean": float(x.mean()),
        "rms": float(np.sqrt(np.mean(x**2))),
        "centered_rms": float(x.std()),
        "positive_authors": int(np.sum(x > 0)),
        "negative_authors": int(np.sum(x < 0)),
        "zero_authors": int(np.sum(x == 0)),
    }


def query_slopes(captures, eta):
    """Return query-specific and differential reader slopes from one BODY direction."""
    x = np.asarray(captures, float)
    delta = secant(x, eta)
    each = (x[:, 1] - x[:, 0]) / (2 * eta)
    np.testing.assert_allclose(delta, each[0] - each[1], atol=1e-12)
    return np.concatenate((each, delta[None]), axis=0)

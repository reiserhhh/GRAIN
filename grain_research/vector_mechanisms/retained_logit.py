"""Exact binary-link accounts and constrained derivative-enhanced response maps."""

from __future__ import annotations

import numpy as np


def logit(p):
    """Invert an interior probability, rejecting saturation rather than clipping it."""
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("Finite probabilities strictly inside (0,1) required")
    return np.log(p) - np.log1p(-p)


def sigmoid(u):
    """Evaluate a finite log-odds value without exponential overflow."""
    u = np.asarray(u, float)
    if not np.isfinite(u).all():
        raise ValueError("Finite log-odds required")
    return np.exp(-np.logaddexp(0, -u))


def link_account(p0, p1, probability_slope):
    """Separate finite probability curvature into link and internal remainders."""
    a, u1 = logit([p0, p1])
    s = float(probability_slope)
    if not np.isfinite(s):
        raise ValueError("Finite directional derivative required")
    b = s / (p0 * (1 - p0))
    projected = float(sigmoid(a + b))
    total = p1 - p0 - s
    link = projected - p0 - s
    internal = p1 - projected
    return {
        "probability_change": p1 - p0,
        "probability_tangent": s,
        "total_remainder": total,
        "link_curvature": link,
        "internal_remainder": internal,
        "logit_change": float(u1 - a),
        "logit_tangent": b,
        "logit_remainder": float(u1 - a - b),
        "identity_error": abs(total - link - internal),
    }


def quadrature(fractions, values):
    """Use precisely the inherited three/five equally spaced Simpson nodes."""
    t, a = np.asarray(fractions, float), np.asarray(values, float)
    if len(t) not in (3, 5) or not np.array_equal(t, np.linspace(0, 1, len(t))):
        raise ValueError("Only existing three/five-node Simpson grids are supported")
    if a.shape[0] != len(t) or not np.isfinite(a).all():
        raise ValueError("Finite values aligned to the integration grid required")
    w = np.ones(len(t))
    w[1:-1:2], w[2:-1:2] = 4, 2
    return np.einsum("i,i...->...", w / (3 * (len(t) - 1)), a)


def path_account(fractions, probabilities, gradients, displacement):
    """Transform cached probability gradients and keep both quadrature errors."""
    p, g, d = map(
        lambda x: np.asarray(x, float), (probabilities, gradients, displacement)
    )
    if (
        p.ndim != 1
        or d.ndim != 2
        or g.shape != (len(p), *d.shape)
        or not np.isfinite(g).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError("Matched finite path x token x hidden arrays required")
    u = logit(p)
    v = g / (p * (1 - p))[:, None, None]
    s = np.einsum("ntd,td->n", g, d)
    b = np.einsum("ntd,td->n", v, d)
    out = link_account(float(p[0]), float(p[-1]), float(s[0]))
    out.update(
        probability_integration_error=float(
            abs(quadrature(fractions, s) - (p[-1] - p[0]))
        ),
        logit_integration_error=float(abs(quadrature(fractions, b) - (u[-1] - u[0]))),
        logit_gradient_delta_norm=float(np.linalg.norm(v[-1] - v[0])),
        logit_gradient_origin_norm=float(np.linalg.norm(v[0])),
        nodes=len(p),
        probability_min=float(p.min()),
        probability_max=float(p.max()),
    )
    return out, v


def radial_parts(values, frame):
    """Resolve role-slot vectors along and orthogonal to a fixed reference frame."""
    x, e = np.asarray(values, float), np.asarray(frame, float)
    if x.shape[-2:] != e.shape or e.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Matched finite slot vectors and frame required")
    if not np.isfinite(e).all() or not np.allclose(np.sum(e**2, axis=1), 1, atol=1e-10):
        raise ValueError("Unit reference frame required")
    parallel = np.sum(x * e, axis=-1, keepdims=True) * e
    return parallel, x - parallel


def fit_curvature(displacements, frames, path_fractions, logit_gradients, penalty=0.01):
    """Fit two scalar Hessian actions per slot from reference gradient increments.

    Each path has equal mass, with equal mass per nonzero node within a path.
    This constrains a potential, not an unrestricted vector-valued regression.
    """
    d, e = np.asarray(displacements, float), np.asarray(frames, float)
    if (
        d.ndim != 3
        or e.shape != d.shape
        or len(path_fractions) != len(d)
        or len(logit_gradients) != len(d)
    ):
        raise ValueError("Aligned nonempty reference paths required")
    if not len(d) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Nonnegative penalty and nonempty paths required")
    numerator, denominator = np.zeros((2, d.shape[1])), np.zeros((2, d.shape[1]))
    origins = []
    for di, ei, ti, vi in zip(d, e, path_fractions, logit_gradients, strict=True):
        ti, vi = np.asarray(ti, float), np.asarray(vi, float)
        quadrature(ti, vi)
        if vi.shape != (len(ti), *di.shape):
            raise ValueError("Gradient dimensions must match displacements")
        origins.append(vi[0])
        x = ti[1:, None, None] * di
        target = vi[1:] - vi[0]
        for k, component in enumerate(radial_parts(x, ei)):
            numerator[k] += np.mean(np.sum(component * target, axis=-1), axis=0) / len(
                d
            )
            denominator[k] += np.mean(np.sum(component**2, axis=-1), axis=0) / len(d)
    coef = np.divide(
        numerator,
        (1 + penalty) * denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.mean(origins, axis=0), coef, denominator


def quadratic_delta(displacement, frame, origin_gradient, coefficients):
    """Return the frozen origin-linear and radial/tangential potential increments."""
    d, v, k = map(
        lambda x: np.asarray(x, float), (displacement, origin_gradient, coefficients)
    )
    if (
        d.shape != v.shape
        or d.ndim != 2
        or k.shape != (2, len(d))
        or not np.isfinite(v).all()
        or not np.isfinite(k).all()
    ):
        raise ValueError("Finite matched potential parameters required")
    parts = radial_parts(d, frame)
    linear = float(np.sum(v * d))
    correction = 0.5 * sum(
        float(np.sum(k[j] * np.sum(part**2, axis=-1))) for j, part in enumerate(parts)
    )
    return linear, linear + correction

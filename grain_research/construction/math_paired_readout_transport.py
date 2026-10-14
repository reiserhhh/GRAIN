"""HM21: paired categorical endpoint actions in a fixed observation law.

The gradient and curvature belong to the readout, not to an author's
psychological response. Coordinates are centered only by the exact logit gauge.
"""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_native_joint as N
from grain_research.construction.math_action_observability import moments


def gradient_hessian(logits, values):
    """Return the exact gradient/Hessian of a fixed categorical expected score."""
    p = N.probability(logits)
    a = np.asarray(values, float)
    if p.ndim != 1 or a.shape != p.shape or not np.isfinite(a).all():
        raise ValueError("one finite categorical endpoint and alphabet required")
    f = float(p @ a)
    g = p * (a - f)
    h = np.diag(g) - p[:, None] * p[None, :] * (a[:, None] + a[None, :] - 2 * f)
    return f, g, h


def paired_action(reference, target, values, order=32):
    """Split an actual endpoint action into local tangent and integrated curvature.

    Per-digit path terms sum to the complete action for this straight segment;
    they do not identify unique internal neural routes.
    """
    r, t = N.center_logits(reference), N.center_logits(target)
    a = np.asarray(values, float)
    if r.ndim != 1 or r.shape != t.shape or order < 2:
        raise ValueError("matched finite endpoints and quadrature order required")
    f, g, _ = gradient_hessian(r, a)
    d = t - r
    y = float(N.score(t, a) - f)
    linear = float(g @ d)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, w = (nodes + 1) / 2, weights / 2
    p = N.probability(r + u[:, None] * d)
    means = p @ a
    gradients = p * (a - means[:, None])
    hessians = np.array([gradient_hessian(z, a)[2] for z in r + u[:, None] * d])
    contraction = np.einsum("j,ijk,k->i", d, hessians, d)
    return {
        "native": y,
        "tangent": linear,
        "curvature": y - linear,
        "gradient_integral": N.response_integral(r, d, a, order=order),
        "hessian_integral": float((w * (1 - u)) @ contraction),
        "reference_score": f,
        "target_score": float(N.score(t, a)),
        "displacement": d,
        "reference_gradient": g,
        "digit_path_action": d * (w @ gradients),
        **N.categorical_geometry(t, r),
    }


def covariance_parts(ya, yb, la, lb, weights):
    """Reconstruct a paired-query covariance retaining signed cross terms."""
    ya, yb, la, lb = (np.asarray(v, float) for v in (ya, yb, la, lb))
    if ya.ndim != 1 or any(v.shape != ya.shape for v in (yb, la, lb)):
        raise ValueError("matched paired-query action vectors required")
    c = moments(np.column_stack((ya, yb, la, lb, ya - la, yb - lb)), weights)[
        "covariance"
    ]
    cross_a, cross_b, cc = float(c[2, 5]), float(c[4, 3]), float(c[4, 5])
    native, ll = float(c[0, 1]), float(c[2, 3])
    link = cross_a + cross_b + cc
    return {
        "native_covariance": native,
        "tangent_covariance": ll,
        "tangent_a_curvature_b": cross_a,
        "curvature_a_tangent_b": cross_b,
        "curvature_covariance": cc,
        "link_covariance_increment": link,
        "recombination_error": abs(native - ll - link),
        "native_tangent_sign_agree": bool(np.sign(native) == np.sign(ll)),
        "link_native_ratio": link / native if native != 0 else None,
    }

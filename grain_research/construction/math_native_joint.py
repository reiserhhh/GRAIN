#!/usr/bin/env python3
"""Categorical-reader interactions in natural and mean coordinates.

Centered logits remove only the exact softmax gauge, not text context. A
declared three-corner baseline separates link curvature from actual logit
interaction; this is an observation law, not a psychological causal model.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def center_logits(logits):
    """Choose the zero-sum representative of the softmax equivalence class."""
    z = np.asarray(logits, float)
    if z.ndim < 1 or z.shape[-1] < 2 or not np.isfinite(z).all():
        raise ValueError("finite categorical logits required")
    return z - z.mean(axis=-1, keepdims=True)


def probability(logits):
    """Return stable categorical probabilities without clipping any answer."""
    z = center_logits(logits)
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


def score(logits, values):
    """Map the fixed conditional answer alphabet to its expected value."""
    p = probability(logits)
    a = np.asarray(values, float)
    if a.shape != (p.shape[-1],) or not np.isfinite(a).all():
        raise ValueError("fixed finite answer values required")
    return p @ a


def interaction(logits, values):
    """Split a 00/10/01/11 contrast at the declared additive-logit endpoint.

    The two terms are signed and can cancel. They are not variance shares,
    unique causal allocations, or fractions of psychological information.
    """
    z = center_logits(logits)
    if z.ndim != 2 or len(z) != 4:
        raise ValueError("complete four-corner categorical table required")
    baseline = z[1] + z[2] - z[0]
    means = score(z, values)
    link_score = float(score(baseline, values))
    return {
        "additive_logits": baseline,
        "logit_interaction": z[3] - baseline,
        "total": float(means[3] - means[1] - means[2] + means[0]),
        "link_only": float(link_score - means[1] - means[2] + means[0]),
        "representation_action": float(means[3] - link_score),
        "link_score": link_score,
    }


def response_integral(origin, displacement, values, order=32):
    """Integrate the exact softmax score gradient along a fixed logit segment."""
    z, d = center_logits(origin), center_logits(displacement)
    if z.ndim != 1 or z.shape != d.shape or order < 2:
        raise ValueError("matched categorical segment and quadrature order required")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    p = probability(z + (nodes[:, None] + 1) * d / 2)
    a = np.asarray(values, float)
    if a.shape != z.shape or not np.isfinite(a).all():
        raise ValueError("fixed answer values required")
    gradients = p * (a - (p @ a)[:, None])
    return float(weights @ (gradients @ d) / 2)


def categorical_geometry(actual, reference):
    """Compute exact KL/Bregman and local Fisher energy in answer units."""
    x, y = center_logits(actual), center_logits(reference)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("matched natural coordinates required")
    p, q = probability(x), probability(y)
    lp, lq = x - logsumexp(x), y - logsumexp(y)
    d = x - y
    return {
        "kl": float(p @ (lp - lq)),
        "bregman": float(logsumexp(y) - logsumexp(x) + p @ d),
        "fisher_energy": float(q @ d**2 - (q @ d) ** 2),
        "centered_displacement_norm": float(np.linalg.norm(d)),
    }


def fit_joint_source(rows):
    """Learn target-specific joint means from source families only."""
    source = [r for r in rows if r["role"] == "source"]
    if not source:
        raise ValueError("source observations required")
    conditional = {}
    for r in source:
        key = (r["frame"], r["actor"], r["incidence"])
        conditional.setdefault(key, []).append(r)
    for group in conditional.values():
        if len({r["family"] for r in group}) != len(group):
            raise ValueError("duplicate source family observation")
    return {
        "global_logits": np.mean([r["logit_interaction"] for r in source], axis=0),
        "conditional": {
            key: {
                "logits": np.mean([r["logit_interaction"] for r in group], axis=0),
                "scalar": float(np.mean([r["total"] for r in group])),
            }
            for key, group in conditional.items()
        },
        "source_families": len({r["family"] for r in source}),
    }


def family_interval(source_effects, application_units, seed=1011, draws=2000):
    """Propagate independent source and application family uncertainty.

    Units contain additive-logit endpoints, observed outcomes and fixed keys.
    Re-estimate conditional means in every draw, never use application targets
    to choose a model or modify a query.
    """
    source = [r for r in source_effects if r["role"] == "source"]
    if not source or any(r["role"] != "application" for r in application_units):
        raise ValueError("disjoint source and application roles required")
    families = sorted({r["family"] for r in source})
    apps = sorted({r["family"] for r in application_units})
    keys = sorted({(r["frame"], r["actor"], r["incidence"]) for r in source})
    lookup = {
        (r["family"], (r["frame"], r["actor"], r["incidence"])): r for r in source
    }
    effects = np.array(
        [[lookup[f, k]["logit_interaction"] for k in keys] for f in families]
    )
    application = {(r["family"], r["key"]): r for r in application_units}
    z = np.array([[application[f, k]["baseline"] for k in keys] for f in apps])
    y = np.array([[application[f, k]["actual"] for k in keys] for f in apps])
    values = np.asarray(application_units[0]["values"], float)
    base = (score(z, values) - y) ** 2
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected_source = rng.integers(0, len(families), len(families))
        selected_app = rng.integers(0, len(apps), len(apps))
        theta = effects[selected_source].mean(axis=0)
        errors = (score(z + theta, values) - y) ** 2
        samples.append(float((base - errors)[selected_app].mean()))
    return np.quantile(samples, [0.025, 0.975]).tolist()

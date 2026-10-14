#!/usr/bin/env python3
"""Ordered three-factor substitutions of a fixed reader's link covector."""

from __future__ import annotations

import itertools
import math

import numpy as np

TERMS = tuple(f"m{i}" for i in range(8))
FIELDS = (
    "full",
    "link",
    "decoder",
    "roundoff",
    "source_mean",
    "source_paired",
    "pairing_loss",
    "delta_link",
    "error_mean",
    "error_paired",
    "error_C",
    "error_H",
    "error_P",
    "error_CH",
    "error_CP",
    "error_HP",
    "full_error_mean",
    "full_error_paired",
    *TERMS,
)


def factors(c, h, p):
    """Validate aligned logit rows without presuming positive curvature."""
    c, h, p = (np.asarray(x, float) for x in (c, h, p))
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("square output curvature required")
    n = len(c)
    if h.ndim < 2 or p.ndim < 2 or h.shape[0] != n or p.shape[0] != n:
        raise ValueError("aligned logit Jacobians required")
    if not all(np.isfinite(x).all() for x in (c, h, p)):
        raise ValueError("finite factors required")
    return c, h.reshape(n, -1), p.reshape(n, -1)


def contract(c, h, p, directions):
    """Keep the physical ordered product J_P^T C J_H d for each H direction."""
    c, h, p = factors(c, h, p)
    d = np.asarray(directions, float)
    if d.ndim < 2 or np.prod(d.shape[1:]) != h.shape[1] or not np.isfinite(d).all():
        raise ValueError("aligned physical H directions required")
    return (d.reshape(len(d), -1) @ h.T) @ c.T @ p


def source_reference(items, roles):
    """Retain two source triples and their separate arithmetic means."""
    if len(items) != 2 or roles != ["source", "source"]:
        raise ValueError("exactly two source wordings required")
    checked = [factors(*x) for x in items]
    if any(a.shape != b.shape for a, b in zip(*checked, strict=True)):
        raise ValueError("source query layouts differ")
    result = {}
    for i, name in enumerate(("c", "h", "p")):
        stack = np.stack([x[i] for x in checked])
        result[name] = stack.mean(axis=0)
        result["source_" + name] = stack
    return result


def crossed(reference, actual, directions):
    """Return all eight substitutions; bits 1/2/4 mean local C/H/P."""
    src = factors(*(reference[k] for k in ("c", "h", "p")))
    act = factors(*actual)
    if any(x.shape != y.shape for x, y in zip(src, act, strict=True)):
        raise ValueError("query-specific relative layouts must align")
    return np.stack(
        [
            contract(
                *(act[j] if mask & (1 << j) else src[j] for j in range(3)), directions
            )
            for mask in range(8)
        ]
    )


def mobius(cube):
    """Invert the full Boolean lattice without selecting an interaction."""
    x = np.asarray(cube, float).copy()
    if x.shape[0] != 8 or not np.isfinite(x).all():
        raise ValueError("all eight finite substitutions required")
    for bit in (1, 2, 4):
        for mask in range(8):
            if mask & bit:
                x[mask] -= x[mask ^ bit]
    return x


def direct_terms(reference, actual, directions):
    """Expand the ordered source-plus-delta factors by direct multilinearity."""
    src = factors(*(reference[k] for k in ("c", "h", "p")))
    act = factors(*actual)
    delta = [x - y for x, y in zip(act, src, strict=True)]
    return np.stack(
        [
            contract(
                *(delta[j] if mask & (1 << j) else src[j] for j in range(3)), directions
            )
            for mask in range(8)
        ]
    )


def paired_source(reference, directions):
    """Average source composite operators, not products of source means."""
    return (
        sum(
            contract(
                *(reference["source_" + k][i] for k in ("c", "h", "p")), directions
            )
            for i in range(2)
        )
        / 2
    )


def moment_summary(rows):
    """Preserve full error moments and all signed component cross moments."""
    if not rows:
        raise ValueError("nonempty observations required")
    n = len(rows)
    result = {"count": n}
    for k in FIELDS:
        x = [float(r[k]) for r in rows]
        if not all(math.isfinite(v) for v in x):
            raise ValueError("finite scalar actions required")
        result[k + "_mean"] = math.fsum(x) / n
        result[k + "_ms"] = math.fsum(v * v for v in x) / n
        result[k + "_rms"] = math.sqrt(result[k + "_ms"])
    components = (*TERMS, "decoder", "roundoff")
    for a, b in itertools.combinations(components, 2):
        result[a + "__" + b + "_cross"] = 2 * math.fsum(r[a] * r[b] for r in rows) / n
    result["full_identity_error"] = max(
        abs(r["full"] - math.fsum(r[k] for k in components)) for r in rows
    )
    result["link_delta_identity_error"] = max(
        abs(r["delta_link"] - math.fsum(r[k] for k in TERMS[1:])) for r in rows
    )
    for variant in ("paired", "C", "H", "P", "CH", "CP", "HP"):
        result["error_" + variant + "_ms_reduction"] = (
            result["error_mean_ms"] - result["error_" + variant + "_ms"]
        )
    return result


def assert_scope(spec):
    """Reject neural work, changed condition scope or an expanded CPU budget."""
    required = {
        "paths": 96,
        "source_paths": 48,
        "source_bundles": 24,
        "maximum_analysis_seconds": 300,
        "maximum_model_calls": 0,
    }
    if any(spec[k] != v for k, v in required.items()):
        raise ValueError("fixed HM41 scope and CPU-only budget required")
    if spec["new_neural_training"] or spec["absolute_tolerance"] != 1e-8:
        raise ValueError("no training or relaxed physical tolerance allowed")

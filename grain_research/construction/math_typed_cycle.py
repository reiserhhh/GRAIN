#!/usr/bin/env python3
"""Complete product-design current sectors; established exterior algebra."""

from __future__ import annotations

import itertools

import numpy as np

NAMES = (
    "category_cycle",
    "level_cycle",
    "category_level",
    "category_interaction",
    "level_interaction",
    "interaction_category_skew",
    "interaction_level_skew",
)


def contrasts(n):
    """Return a fixed orthonormal Helmert basis of the centered node space."""
    if not isinstance(n, int) or n < 3:
        raise ValueError("At least three discrete categories required")
    a = np.zeros((n, n - 1))
    for j in range(1, n):
        a[:j, j - 1] = 1 / np.sqrt(j * (j + 1))
        a[j, j - 1] = -j / np.sqrt(j * (j + 1))
    return a


def squares(n, skew):
    """Construct Frobenius-orthonormal symmetric OR alternating matrices."""
    if n < 2:
        raise ValueError("Square dimensions must be at least two")
    out = []
    if not skew:
        out.extend(np.diag(np.eye(n)[i]) for i in range(n))
    for i, j in itertools.combinations(range(n), 2):
        a = np.zeros((n, n))
        a[i, j], a[j, i] = 1 / np.sqrt(2), (-1 if skew else 1) / np.sqrt(2)
        out.append(a)
    return np.stack(out, axis=-1)


def sectors(categories=6, levels=3):
    """Return all seven current bases, never selected using target outcomes."""
    b, l = contrasts(categories), contrasts(levels)
    eb, el = (
        np.ones((categories, 1)) / np.sqrt(categories),
        np.ones((levels, 1)) / np.sqrt(levels),
    )
    nodes = (np.kron(b, el), np.kron(eb, l), np.kron(b, l))
    out = {}
    for name, first, second in zip(
        NAMES[:5], (0, 1, 0, 0, 1), (0, 1, 1, 2, 2), strict=True
    ):
        x, y = nodes[first], nodes[second]
        pairs = (
            itertools.combinations(range(x.shape[1]), 2)
            if first == second
            else itertools.product(range(x.shape[1]), range(y.shape[1]))
        )
        out[name] = np.stack(
            [
                (np.outer(x[:, i], y[:, j]) - np.outer(y[:, j], x[:, i])) / np.sqrt(2)
                for i, j in pairs
            ],
            axis=-1,
        )
    for name, skew_b in zip(NAMES[5:], (True, False), strict=True):
        xb, xl = squares(categories - 1, skew_b), squares(levels - 1, not skew_b)
        matrices = []
        for i, j in itertools.product(range(xb.shape[2]), range(xl.shape[2])):
            local = np.kron(xb[..., i], xl[..., j])
            matrices.append(nodes[2] @ local @ nodes[2].T)
        out[name] = np.stack(matrices, axis=-1)
    return out


def project(kernel, bases):
    """Orthogonally account for every closed-current component of a kernel."""
    f = np.asarray(kernel, float)
    n = next(iter(bases.values())).shape[0]
    if f.ndim != 3 or f.shape[:2] != (n, n) or not np.isfinite(f).all():
        raise ValueError("Complete finite square pair kernel with channels required")
    coefficients, parts = {}, {}
    for name, a in bases.items():
        coefficients[name] = np.einsum("ija,ijq->aq", a, f)
        parts[name] = np.einsum("ija,aq->ijq", a, coefficients[name])
    return coefficients, parts


def positive_laws(bases):
    """Build all basis currents and two positive equal-marginal laws per one."""
    d = np.concatenate(list(bases.values()), axis=-1).transpose(2, 0, 1)
    n = d.shape[1]
    scale = 1 / (2 * n**2 * np.max(np.abs(d), axis=(1, 2)))
    base = np.full((n, n), 1 / n**2)
    return d, base + scale[:, None, None] * d, base - scale[:, None, None] * d


def compare(source, application):
    """Return all geometric accounts without fitting a scale or hiding zero."""
    x, y = np.asarray(source, float), np.asarray(application, float)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite([x, y]).all():
        raise ValueError("Aligned finite current coefficient matrices required")
    sx, sy = np.sum(x * x, 0), np.sum(y * y, 0)
    dot, defect = np.sum(x * y, 0), np.sum((x - y) ** 2, 0)
    cosine = np.divide(
        dot, np.sqrt(sx * sy), out=np.full_like(dot, np.nan), where=(sx > 0) & (sy > 0)
    )
    return {
        "source_energy": sx,
        "application_energy": sy,
        "cross": dot,
        "defect_energy": defect,
        "cosine": cosine,
    }

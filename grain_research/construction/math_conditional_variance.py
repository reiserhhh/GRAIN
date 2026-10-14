#!/usr/bin/env python3
"""Exact functional ANOVA under an explicitly uniform finite product design.

This is a diagnostic of a controlled design, not a decomposition of correlated
natural tokens or a claim that removed terms are psychological noise.
"""

from __future__ import annotations

import numpy as np


def product_components(values):
    """Return all orthogonal conditional components and reconstruction checks.

    Every tensor axis is an independently crossed, uniformly weighted design
    factor. Missing or nonfinite cells are rejected rather than imputed.
    Component zero is the mean. Other components have zero conditional mean
    on each of their included axes.
    """
    f = np.asarray(values, dtype=float)
    if not f.ndim or any(n < 2 for n in f.shape) or not np.isfinite(f).all():
        raise ValueError("require a finite complete product with >=2 levels per axis")
    components = {}
    for mask in range(1 << f.ndim):
        absent = tuple(j for j in range(f.ndim) if not mask & (1 << j))
        conditional = f.mean(axis=absent, keepdims=True) if absent else f.copy()
        term = conditional.copy()
        for previous, component in components.items():
            if previous & mask == previous:
                term = term - component
        components[mask] = term
    expanded = [np.broadcast_to(v, f.shape) for v in components.values()]
    reconstruction = np.sum(expanded, axis=0)
    energies = {m: float(np.mean(v**2)) for m, v in components.items()}
    gram = np.asarray(expanded).reshape(len(expanded), -1)
    gram = gram @ gram.T / f.size
    off_diagonal = gram - np.diag(np.diag(gram))
    variance = float(np.var(f))
    return {
        "components": components,
        "energies": energies,
        "mean": float(f.mean()),
        "variance": variance,
        "mse": float(np.mean(f**2)),
        "max_reconstruction_error": float(np.max(np.abs(reconstruction - f))),
        "max_orthogonality_error": float(np.max(np.abs(off_diagonal))),
        "variance_closure_error": abs(
            sum(v for m, v in energies.items() if m) - variance
        ),
    }
